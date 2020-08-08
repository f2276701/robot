#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from collections import deque
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from rnn import LSTMPB, GRUPB
from utils import load_model, load_data, make_fig


WINDOW_SIZE =50 

class HistoryWindow(object):
    def __init__(self, *args, **kwargs):
        self.deque = deque(*args, **kwargs)

    def put(self, item):
        self.deque.append(item)

    def get(self):
        prediction, actual, states = [], [], []
        while len(self.deque):
            p, a, s = self.deque.popleft()
            prediction.append(p)
            actual.append(a)
            states.append(s)
        return torch.stack(prediction, 1), torch.stack(actual, 1), states


def main(args):
    ### Loading Dataset ###
    data_root = os.path.join(args.data_path, "train")
    dataset, _ = load_data(data_root)
    pb_list = torch.from_numpy(np.loadtxt(os.path.join(args.data_path, "train", "pb_list.txt"), 
                                          dtype=np.float32, delimiter=",").reshape(-1, args.pb_dims))
    rnn = LSTMPB(args, pb_unit=pb_list[None, 0])
    #rnn = LSTMPB(args, pb_unit=nn.Parameter(pb_list[None, 0],
    #                                        requires_grad=True))
    data = dataset[None, 0]

    pt_file = load_model(args.model_load_path, "*.pt")
    state_dict = torch.load(pt_file)
    rnn.load_state_dict(state_dict)
    for param in rnn.parameters():
        param.requires_grad = False

    rnn.pb_unit = nn.Parameter(pb_list[None, 0], requires_grad=True)
    optim_param = [param for param in rnn.parameters() if param.requires_grad == True]
    print(optim_param)
    optim = torch.optim.Adam(optim_param, lr=0.01)
    #optim = torch.optim.Adam(optim_param)
    mse_loss = nn.MSELoss()
    his_log = HistoryWindow(maxlen=WINDOW_SIZE)
    #rnn.eval()

    output_log = []
    print(rnn.pb_unit)
    pb_log = []
    for i in range(data.shape[1] - 1):
        if i == 0:
            cur_input = data[:, 0, :]
            state = None
        else:
            if i == 999:
                pred_his, actual_his, state_his = his_log.get()
                for _ in range(100):
                    log = []
                    cur = torch.cat([pred_his[:, 0, :5], actual_his[:, 0, 5:]], dim=-1)
                    s = state_his[0]
                    for step in range(1, len(state_his)):
                        o, prev_s = rnn.step(cur, s)
                        cur = o
                        s = prev_s
                        log.append(o)
                        
                    log = torch.stack(log, dim=1)
                    #loss = mse_loss(log[0, :, 5:], actual_his[0, 1:, 5:]) + (rnn.pb_unit - pb_list).pow(2).mean()
                    loss = mse_loss(log[0, :, 5:], actual_his[0, 1:, 5:]) + (rnn.pb_unit - pb_list).pow(2).mean()
                    loss.backward(retain_graph=True)
                    pb_log.append(rnn.pb_unit.data.clone())
                    optim.step()
                    print(loss.item())

            #    rnn.pb_unit=pb_list[None, 5]
            #    data = dataset[None, 5]
                prev_state = s


            cur_input = torch.cat([output[:, :5], data[:, i, 5:]], dim=-1)
            #if i < 100:
            #    cur_input = torch.cat([output[:, :5], data[:, i, 5:]], dim=-1)
            #else:
            #    #cur_input = torch.cat([output[:, :5], dataset[0,None, i, 5:]], dim=-1)
            #    cur_input = output
            #cur_input = data[:, i, :]
            state = prev_state

        output, prev_state = rnn.step(cur_input, state)
        his_log.put([output[:, :], data[:, i + 1, :], prev_state])
        output_log.append(output.detach())

    print(rnn.pb_unit)
    output_log = torch.stack(output_log, dim=1)
    np.savetxt("{}LSTMPB_closed_predict.txt".format(args.log_path), 
               output_log[0].numpy(), delimiter=",")
    fig = make_fig([[output_log[0], dataset[0]]], figsize=(16, 16))
    fig.savefig("{}LSTMPB_closed_predict_goal000.png".format(args.log_path))
    plt.close(fig)
    fig = make_fig([[output_log[0], dataset[5]]], figsize=(16, 16))
    fig.savefig("{}LSTMPB_closed_predict_goal001.png".format(args.log_path))
    plt.close(fig)
    fig = make_fig([[output_log[0], dataset[10]]], figsize=(16, 16))
    fig.savefig("{}LSTMPB_closed_predict_goal002.png".format(args.log_path))
    plt.close(fig)
    fig = make_fig([[output_log[0], dataset[15]]], figsize=(16, 16))
    fig.savefig("{}LSTMPB_closed_predict_goal003.png".format(args.log_path))
    plt.close(fig)

    pb_log = torch.stack(pb_log, 1)
    np.savetxt("{}pb_log.txt".format(args.log_path), 
               pb_log[0].numpy(), delimiter=",")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("--unit_nums", type=int, default=100)
    parse.add_argument("--in_dims", type=int, default=12)
    parse.add_argument("--out_dims", type=int, default=12)
    parse.add_argument("--closed_index", type=int, default=5)
    parse.add_argument("--closed_rate", type=float, default=1)
    parse.add_argument("--pb_dims", type=int, default=7)
    parse.add_argument("--data_path", type=str, default="./data/20200706/")
    #parse.add_argument("--model_load_path", type=str, default="./result/result200611_RMSPROB_closed_succeed/LSTMPB/model/")
    #parse.add_argument("--model_load_path", type=str, default="./result/result200618_17/LSTMPB/model/")
    #parse.add_argument("--model_load_path", type=str, default="./result/result200618/finetuning_LSTMPB/model/")
    parse.add_argument("--model_load_path", type=str, default="./result/result200708_22/LSTMPB/model/")
    #parse.add_argument("--log_path", type=str, default="./result/result200611_RMSPROB_closed_succeed/LSTMPB/")
    #parse.add_argument("--log_path", type=str, default="./result/result200618_17/LSTMPB/")
    #parse.add_argument("--log_path", type=str, default="./result/result200618/finetuning_LSTMPB/")
    parse.add_argument("--log_path", type=str, default="./result/result200708_22/LSTMPB/")
    args = parse.parse_args()

    main(args)
