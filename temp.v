module fxp2mf # (
    parameter EBIT = 2,
    parameter MBIT = 1,
    parameter bm_width = 4,
    parameter data_width = 16,
    parameter BIAS = 1
) (
  //input  wire                        clk,
  //input  wire                        rst_n,
  input  wire  [7:0]                current_ps,
  input  wire  [data_width-1:0]     up_fxp,
  //input  wire                       up_vld,  
  //output wire                       dn_vld,
  output wire  [bm_width-1:0]       dn_dat
);

    localparam ADD_OFFSET = 0;

    wire [data_width-1:0] fxp;
    wire [$clog2((data_width + 1) + ADD_OFFSET) - 1:0] productClz;

    assign fxp = ( up_fxp[data_width-1]) ? ~(up_fxp - 1'b1) : up_fxp;
    
    CountLeadingZeros #(.WIDTH(data_width))
        clz(.in(fxp),
            .out(productClz));

    wire signed [EBIT+1:0] exp;
    assign exp = data_width - productClz - current_ps - 1;  // is signed exp

    reg [MBIT-1:0] man;
    reg [EBIT-1:0] exp_out;

    always @(*) begin
        if (exp < -BIAS) begin 
        // flush to zero
            man <= 0;
            exp_out <= 0;
        end
        else if (exp == -BIAS) begin
        // denormal
        // if (ACC_OUT_LEN-productClz <= current_ps)
            man <= fxp[(data_width-productClz-MBIT)+:MBIT];
            exp_out <= 0;
        end
        else begin
        // normal 
            man <= fxp[(data_width-productClz-MBIT-1)+:MBIT];
            exp_out <= exp + BIAS;
        end
    end

    assign dn_dat = { up_fxp[data_width-1], exp_out, man};

endmodule


module bm_renorm
# (
  parameter data_width = 16,
  parameter bm_width = 4,
  parameter num_output = 8
)
(
  input  wire                        clk,
  input  wire                        rst_n,
  input  wire                        bn_enable,

  input  wire  [num_output * data_width-1:0]  up_dat,
  input  wire                        up_vld,
  input  wire  [16-1:0]              length,
  
  output wire                        dn_vld,
  output wire  [num_output * bm_width-1:0]    dn_dat
);

    localparam block_len = length / num_output; // fix later
    reg [num_output * data_width-1:0] fxp_mem [block_len-1:0];

    genvar i,j;

    reg store_done;
    reg [:] counter;
    reg [:] mask_cnt;

    always @(posedge clk or negedge rst_n)
        if(!rst_n) begin
            // todo all zero
            fxp_mem[:] <= 128'b0;
            counter <= 0;
            store_done <= 0;
        end
        else begin
            if (up_vld) begin
                fxp_mem[counter] <= up_dat;
                mask_cnt <= counter;
                
                if (counter == (block_len-1)) begin    
                    counter <= 0;
                    store_done <= 1'b1;

                end else begin
                    counter <= counter + 1;
                    store_done <= 0;
                end
            end
        end

    wire unsigned [data_width-1:0] mask_array [block_len:0];
    wire [data_width-1:0] temp [7:0];

    generate
        for (i=0;i<block_len;i=i+1) begin
            assign temp[0] = fxp_mem[i][data_width*8-1] ?
                             ~(fxp_mem[i][(data_width*8-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*8-1)-:data_width];
            assign temp[1] = fxp_mem[i][data_width*7-1] ?
                             ~(fxp_mem[i][(data_width*7-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*7-1)-:data_width];
            assign temp[2] = fxp_mem[i][data_width*6-1] ?
                             ~(fxp_mem[i][(data_width*6-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*6-1)-:data_width];
            assign temp[3] = fxp_mem[i][data_width*5-1] ?
                             ~(fxp_mem[i][(data_width*5-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*5-1)-:data_width];
            assign temp[4] = fxp_mem[i][data_width*4-1] ?
                             ~(fxp_mem[i][(data_width*4-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*4-1)-:data_width];
            assign temp[5] = fxp_mem[i][data_width*3-1] ?
                             ~(fxp_mem[i][(data_width*3-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*3-1)-:data_width];
            assign temp[6] = fxp_mem[i][data_width*2-1] ?
                             ~(fxp_mem[i][(data_width*2-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*2-1)-:data_width];
            assign temp[7] = fxp_mem[i][data_width*1-1] ?
                             ~(fxp_mem[i][(data_width*1-1)-:data_width]-1'b1) : fxp_mem[i][(data_width*1-1)-:data_width];

            assign mask_array[i] = temp[0] | temp[1] | temp[2] | temp[3] | temp[4] | temp[5] | temp[6] | temp[7];
        end
    endgenerate

    reg [data_width-1:0] mask_r;

    always @(posedge clk or negedge rst_n)
        if(!rst_n) begin
            mask_r <= 0;
        end
        else begin
            mask_r <= mask_r | mask_array[mask_cnt];
            if (store_done) begin
                
            end else begin
                mask_r <= mask_r;
            end
        end

    localparam ADD_OFFSET = 0;
    wire [$clog2((data_width + 1) + ADD_OFFSET) - 1:0] productClz;

    CountLeadingZeros #(.WIDTH(data_width))
        clz(.in(mask_r),
            .out(productClz));

    localparam implicit_dot = 0;
    wire signed [7:0] shift;
    wire signed [7:0] satu_sft;
    wire signed [7:0] current_ps;

    assign shift = data_width - productClz - implicit_dot;
    assign satu_sft = shift - EMAX - 1; // this value will be added to the external shared exponent.
    assign current_ps = implicit_dot + satu_sft;

    wire [bm_width-1:0] dn_dats_all [];

    generate
        for (i = 0; i < block_len; i = i + 1) begin
            for (j = 0; j < num_output; j = j + 1) begin

                fxp2mf # (
                    .EBIT(EBIT),
                    .MBIT(MBIT),
                    .BIAS(BIAS),
                    .bm_width(bm_width),
                    .data_width(data_width)
                ) fxp2mf_convertor (
                    .current_ps(current_ps),
                    .up_fxp(fxp_mem[i][(j*data_width)+:data_width]),
                    .dn_dat(dn_dats_all[i][(j*bm_width)+:bm_width])
                );
            end
        end
    endgenerate


endmodule
