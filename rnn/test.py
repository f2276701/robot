#!/usr/bin/env python
#coding:utf-8

'''
Created on
Update  on
Author: JumpLion
Team:
Github: https://f2276701.github.io/
'''

from __future__ import print_function

import sys
import glob
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils import sampling, DataGenerator

from rnn import LSTMPB, LSTM
from _rnn import LSTM
from utils import load_model, load_data, make_fig


def main(args):
    ### Loading Dataset ###
    data_root = os.path.join(args.data_path, "train")
    dataset, _ = load_data(data_root)

    rnn = LSTM(args)
    data = dataset[None, 0]

    pt_file = load_model(args.model_load_path, "*.pt")
    rnn.load_state_dict(torch.load(pt_file))
    rnn.eval()

    output_log = []
    for i in range(data.shape[1] - 1):
        if i == 0:
            cur_input = data[:, 0, :]
            state = None
        else:
            cur_input = torch.cat([output[0, :, :8], data[:, i, 8:]], dim=-1)
            #cur_input = data[:, i, :]
            state = prev_state

        output, prev_state = rnn(cur_input.view(1, 1, -1), state)
        output_log.append(output.detach())

    output_log = torch.stack(output_log, dim=1)
    np.savetxt("{}LSTMPB_closed_predict.txt".format(args.log_path), 
               output_log[0, 0].numpy(), delimiter=",")
    fig = make_fig([[output_log[0, :, 0, :], dataset[1]]], figsize=(16, 16))
    fig.savefig("{}LSTMPB_closed_predict.png".format(args.log_path))
    plt.close(fig)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--unit_nums", type=int, default=100)
    parse.add_argument("--in_dims", type=int, default=11)
    parse.add_argument("--out_dims", type=int, default=11)
    parse.add_argument("--closed_index", type=int, default=8)
    parse.add_argument("--closed_rate", type=float, default=1)
    parse.add_argument("--pb_dims", type=int, default=3)
    parse.add_argument("--data_path", type=str, default="./data/20200618_2/")
    #parse.add_argument("--model_load_path", type=str, default="./result/result200611_RMSPROB_closed_succeed/LSTMPB/model/")
    #parse.add_argument("--model_load_path", type=str, default="./result/result200618_17/LSTMPB/model/")
    #parse.add_argument("--model_load_path", type=str, default="./result/result200618/finetuning_LSTMPB/model/")
    parse.add_argument("--model_load_path", type=str, default="./result/result200618/LSTM/model/")
    #parse.add_argument("--log_path", type=str, default="./result/result200611_RMSPROB_closed_succeed/LSTMPB/")
    #parse.add_argument("--log_path", type=str, default="./result/result200618_17/LSTMPB/")
    #parse.add_argument("--log_path", type=str, default="./result/result200618/finetuning_LSTMPB/")
    parse.add_argument("--log_path", type=str, default="./result/result200618/LSTM/")
    args = parse.parse_args()

    main(args)
