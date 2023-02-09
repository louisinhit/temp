`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Design Name: 
// Module Name: butterfly_p2s_ln_opt
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module butterfly_p2s_ln_opt
# (
  parameter data_width = 16,
  parameter num_output = 8,
  parameter bm_width = 4,
  parameter EBIT = 2,
  parameter MBIT = 1,
  parameter BIAS = 1
)
(
  input  wire                             clk,
  input  wire                             rst_n,
  input  wire                             is_ln,
  input  wire  [16-1:0]                   length,
  input  wire  [num_output*data_width-1:0]  up_dat,
  input  wire                             up_vld,
  input  wire                             by_pass,

  output wire                             up_rdy,
  output wire  [num_output*bm_width-1:0]  dn_parallel_dat,
  output wire                             dn_parallel_vld,
  input wire                              dn_parallel_rdy,
  output wire  [bm_width-1:0]             dn_serial_dat,
  output wire                             dn_serial_vld,
  input wire                              dn_serial_rdy,
  output wire signed [7:0]                exp_shift
);


wire  [num_output*data_width-1:0]  up_ln_dat;
wire                        up_ln_vld;

wire  [num_output*data_width-1:0]  dn_ln_dat;
wire                        dn_ln_vld;
wire                        dn_ln_rdy;

assign up_ln_dat = is_ln? up_dat : 0;
assign up_ln_vld = is_ln? up_vld : 0;

layer_norm #(
    .data_width(data_width),
    .p_ln(num_output)
) u_layer_norm
(
    .rst_n(rst_n),
    .clk(clk),

    .up_vld(up_ln_vld),
    .up_dat(up_ln_dat),
    .up_rdy(),
    
    .bias_ln(16'b0), // Assume bias is zero
    .length(length),

    .dn_vld(dn_ln_vld),
    .dn_rdy(dn_ln_rdy),
    .dn_dat(dn_ln_dat)
);


wire  [num_output*data_width-1:0]  up_bm_dat;
wire                               up_bm_vld;
wire  [num_output*bm_width-1:0]    dn_bm_dat;
wire                               dn_bm_vld;

assign up_bm_dat = is_ln? dn_ln_dat : up_dat;
assign up_bm_vld = is_ln? dn_ln_vld : up_vld;

bm_renorm #(
  .data_width(data_width),
  .bm_width(bm_width),
  .num_output(num_output),
  .EBIT(EBIT),
  .MBIT(MBIT),
  .BIAS(BIAS)
) u_bm_renorm
(
  .clk(clk),
  .rst_n(rst_n),

  .up_dat(up_bm_dat),
  .up_vld(up_bm_vld),
  .length(length),
  
  .dn_vld(dn_bm_vld),
  .dn_dat(dn_bm_dat),
  .satu_sft(exp_shift)
);

/*
wire  [bm_output*data_width-1:0]  up_p2s_dat;
wire                              up_p2s_vld;

assign up_p2s_dat = is_bm? dn_bm_dat : up_bm_dat;
assign up_p2s_vld = is_bm? dn_bm_vld : up_bm_vld;
*/

butterfly_p2s_opt # (
  .data_width(bm_width),
  .num_output(num_output)
) u_butterfly_p2s
(
  .clk(clk),
  .rst_n(rst_n),
  .by_pass(by_pass),
  .up_dat(dn_bm_dat),
  .up_vld(dn_bm_vld),
  .up_rdy(dn_ln_rdy),
  .dn_parallel_dat(dn_parallel_dat),
  .dn_parallel_vld(dn_parallel_vld),
  .dn_parallel_rdy(dn_parallel_rdy),
  .dn_serial_dat(dn_serial_dat),
  .dn_serial_vld(dn_serial_vld),
  .dn_serial_rdy(dn_serial_rdy)
);

endmodule
