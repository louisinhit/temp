import torch
import numpy as np
import os
import argparse
#################### Get from https://github.com/HazyResearch/butterfly/blob/master/torch_butterfly/multiply.py###########
import math

from torch.nn import functional as F

import random
from quant import *
from fxpmath import Fxp

code = {}

code['0000'] = 0.
code['0001'] = 0.5
code['0010'] = 1.0
code['0011'] = 1.5

code['0100'] = 2.
code['0101'] = 3.
code['0110'] = 4.
code['0111'] = 6.

code['1000'] = -0.
code['1001'] = -0.5
code['1010'] = -1.0
code['1011'] = -1.5

code['1100'] = -2.
code['1101'] = -3.
code['1110'] = -4.
code['1111'] = -6.


code_list = [code[i] for i in code.keys()]


def float_to_bits(num, exp_b, sig_b):
    sign = ('1' if num<0 else '0')
    num = abs(num)
    bias = (2**(exp_b-1)-1)
    e = (0 if num==0 else math.floor(math.log(num,2)+bias))
    if(e>(2**(exp_b)-1)):#overflow
        exponent = '1'*exp_b
        mantissa = '0'*sig_b
    else:
        if(e>0):#normal
            s = num/2**(e-bias)-1
            exponent = bin(e)[2:].zfill(exp_b)
        else:   #submoral
            s = num/2**(-bias+1)
            exponent = '0'*exp_b
        mantissa = bin(int(s*(2**sig_b)+0.5))[2:].zfill(sig_b)[:sig_b]
    return sign+exponent+mantissa


def float_to_bm(num_list):
    out_list = []
    for i in num_list:
        out_list.append(float_to_bits(num=i, exp_b=2, sig_b=1))
    return np.asarray(out_list)


def butterfly_multiply_torch(twiddle, input, increasing_stride=True, output_size=None):
    batch_size, nstacks, input_size = input.shape
    nblocks = twiddle.shape[1]
    log_n = twiddle.shape[2]
    n = 1 << log_n
    assert twiddle.shape == (nstacks, nblocks, log_n, n // 2, 2, 2)
    # Pad or trim input to size n
    input = F.pad(input, (0, n - input_size)) if input_size < n else input[:, :, :n]
    output_size = n if output_size is None else output_size
    assert output_size <= n
    output = input.contiguous()
    cur_increasing_stride = increasing_stride
    intern_results = []
    weights = []
    for block in range(nblocks):
        for idx in range(log_n):
            log_stride = idx if cur_increasing_stride else log_n - 1 - idx
            stride = 1 << log_stride
            # shape (nstacks, n // (2 * stride), 2, 2, stride)
            # Get weights data
            tmp_weight = twiddle[:, block, idx].clone()
            tmp_weight = tmp_weight.view(nstacks, n // 2, 2, 2).permute(0, 1, 3, 2) 
            weights.append(tmp_weight.reshape(nstacks, n // 2, 2*2))
            t = twiddle[:, block, idx].view(
                nstacks, n // (2 * stride), stride, 2, 2).permute(0, 1, 3, 4, 2)
            output_reshape = output.view(
                batch_size, nstacks, n // (2 * stride), 1, 2, stride)
            output = (t * output_reshape).sum(dim=4)
            intern_results.append(output.view(batch_size, nstacks, n))
        cur_increasing_stride = not cur_increasing_stride
    return output.view(batch_size, nstacks, n)[:, :, :output_size], intern_results, weights

######################################

def get_offset(length, bram_width=4):
    depth = length/bram_width
    offset = [0, 1]
    depth -= 2
    while (depth>0):
        depth -= len(offset)
        new_offset = [(i+1)%bram_width for i in offset]
        offset = offset + new_offset
    return offset

'''
Due the bank conflict of butterfly, weights need to be rearranged.
As weights are fixed during the inference, this process is performed offline.
'''
def reorder_weight(weights, length, bu_parallelism):
    bram_width = 8
    evens = list(range(0, bram_width, 2))
    odds = list(range(1, bram_width, 2))
    offset = get_offset(length, bram_width)
    num_stage = len(weights)
    for i in range(num_stage):
        weight = weights[i]
        if (num_stage - i) > math.log2(bram_width): # Last two stage no need to reorder
            stride = (1 << (num_stage-i)) // bram_width
            depth = length // bram_width
            cur_d = 0
            seq = []
            while (cur_d<depth):
                for j in range(stride//2):
                    relative_seq = evens + odds
                    abs_seq = [s + len(seq) for s in relative_seq]
                    seq = seq + abs_seq
                cur_d += stride
            seq = torch.tensor(seq)
            weight = torch.squeeze(weight, 0)
            weight = weight[seq]
            weights[i] = torch.unsqueeze(weight, 0)
        # (nstacks, n // 2, 2*2)
        weight_shape = weights[i].shape
        weights[i] = weights[i].view(weight_shape[0], weight_shape[1]//bu_parallelism, weight_shape[2]*bu_parallelism)


def gen_bfly_float16(args):
    n = args.length # bfly_length
    log_n = int(math.ceil(math.log2(n)))
    bu_parallelism = 4

    twiddle_shape = (1, 1, log_n, n // 2, 2, 2)
    ########## new twiddle...
    twiddle = random.choices(code_list, k=int(1*1*log_n*(n // 2)*2*2))
    twiddle = torch.tensor(twiddle).reshape(twiddle_shape)
    ##################

    batch_size = 1
    input_shape = (batch_size, 1, n)
    #inputs = torch.rand(input_shape)
    mf_inputs = random.choices(code_list, k=int(batch_size*1*n))
    mf_inputs = torch.tensor(mf_inputs).reshape(input_shape)
    
    inputs = mf_inputs

    print ("========Running Butterfly=======")
    _, intern_results, weights = butterfly_multiply_torch(twiddle, inputs, increasing_stride=False)
    reorder_weight(weights, n, bu_parallelism)


    path = './float16_bfly'+str(n)
    print ("Gnerating test data")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get Initial Input
    np_input = torch.squeeze(mf_inputs).cpu().detach().numpy()
    np_input = float_to_bm(np_input)
    print (np_input)
    
    np.savetxt(path+'/input_bfly.txt', np_input, delimiter='\n', fmt='%s')

    # Get Intern Result and Weight
    for i in range(len(intern_results)):
        #np.savetxt(path+'/data_stage'+str(i)+'.txt', (torch.squeeze(intern_results[i]).cpu().detach().numpy()).astype(np.float16), delimiter='\n', fmt='%s')
        #######################
        x = torch.squeeze(weights[i]).cpu().detach().numpy().flatten()
        x = float_to_bm(x)
        np.savetxt(path+'/weight_stage'+str(i)+'.txt', x, delimiter='\n', fmt='%s')
    
    num = BlockMinifloat(exp=2, man=1, tile=0)
    qdata, se = block_minifloat_quantize(torch.squeeze(intern_results[-1]).cpu().detach(), number=num, rounding="nearest", tensor_type="x")
    print ("output shared exponent is : ", se)

    #p = Fxp(torch.squeeze(intern_results[-1]).cpu().detach().numpy(), signed=True, n_word=int(32), n_frac=8)
    #for ii in p.bin():
    #    print ("output in fxp : ", ii)

    for j in qdata:
        x = float_to_bits(num=j, exp_b=2, sig_b=1)
        print (j)
        with open('output_bfly.txt', 'a') as f:
            f.write(f'{j}\n')

    print ("========Done=======")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    f = open('mat_out.txt', 'w')

    print ("code list : ", code_list)
    parser.add_argument("--length", type = int, help = "length of sequence", default = 256)
    args = parser.parse_args()
    gen_bfly_float16(args)

    for i in code.keys():
        print ('number is :', code[i])
        print ('correct bin :', i, 'convertor out :', float_to_bits(num=code[i], exp_b=2, sig_b=1))
        
