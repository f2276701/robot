#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import Model
from rnn import RNN, RNNPB, LSTM, LSTMPB, GRU, GRUPB
import config
from utils import load_model, make_fig, load_data


def main(args):
    #########################################
    ###          Loading Dataset          ###
    #########################################
    train_root = os.path.join(args.data_path, "train")
    train_data, train_loader = load_data(train_root)
    test_data, test_loader = train_data, train_loader

    model_args = {"args":args}
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if "PB" in args.net:
        train_pb_list = np.loadtxt(os.path.join(args.data_path, "train", "pb_list.txt"), 
                                   dtype=np.float32, delimiter=",").reshape([-1, args.pb_dims])
        train_pb_list = torch.from_numpy(train_pb_list).to(device)
        test_pb_list = train_pb_list
        model_args["pb_unit"] = train_pb_list

    if args.validate:
        vali_root = os.path.join(args.data_path, "validation")
        vali_data, vali_loader = load_data(vali_root)
        if "PB" in args.net:
            vali_pb_list = np.loadtxt(os.path.join(args.data_path, "validation", "pb_list.txt"), 
                                      dtype=np.float32, delimiter=",")
            vali_pb_list = torch.from_numpy(vali_pb_list).to(device).to(device)
    else:
        vali_loader, vali_pb_list = None, None

    ### Building Model ###
    model = Model(args)
    model.build(eval(args.net), **model_args)

    if args.model_load_path:
        model_name = args.net + "*.pt"
        pt_file = load_model(args.model_load_path, model_name)
        model.load(pt_file)

    if args.mode == "train":
        model.rnn.train()
        model.train(train_loader, vali_loader, vali_pb_list)
    elif args.mode == "test":
        model.rnn.eval()
        output_log, state_log, loss = model.test(test_loader)
        print("test loss:", loss)
        for i in range(output_log.shape[0]):
            fig = make_fig([[output_log[i], test_data[i].numpy()]])
            np.savetxt("{}{}_test_{:0>6d}.txt".format(args.log_path, args.net, i), 
                       output_log[i], delimiter=" ")
            plt.savefig("{}{}_test_{:0>6d}.png".format(args.log_path, args.net, i))
            plt.close(fig)
    elif args.mode == "predict":
        model.rnn.eval()
        inputs = test_data.to(device)
        output_log, state_log = model.predict(inputs, None, inputs.shape[1] - 1)
        for i in range(test_data.shape[0]):
            fig = make_fig([[output_log[i], test_data[i].numpy()]])
            np.savetxt("{}{}_predict_{:0>6d}.txt".format(args.log_path, args.net, i), 
                       output_log[i], delimiter=" ")
            fig.savefig("{}{}_predict_{:0>6d}.png".format(args.log_path, args.net, i))
            plt.close(fig)


if __name__ == "__main__":
    args = config.get_config()
    main(args)

