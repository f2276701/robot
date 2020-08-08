#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def get_config():
    parse = argparse.ArgumentParser()

    # Moel config
    parse.add_argument("--unit_nums", type=int, default=32)
    parse.add_argument("--in_dims", type=int, default=2)
    parse.add_argument("--out_dims", type=int, default=2)
    parse.add_argument("--closed_index", type=int, default=5)
    parse.add_argument("--closed_rate", type=float, default=0.5)
    parse.add_argument("--pb_dims", type=int, default=2)

    # Training config
    parse.add_argument("--batch_size", type=int, default=100)
    parse.add_argument("--lr", type=float, default=1e-3)
    parse.add_argument("--delay", type=int, default=1)
    parse.add_argument("--epoch", type=int, default=1000)
    parse.add_argument("--closed_step", type=int, default=700)
    parse.add_argument("--validate", type=bool, default=False)
    parse.add_argument("--cuda", type=bool, default=True)
    parse.add_argument("--tensorboard", type=bool, default=False)

    #parse.add_argument("--data_path", type=str, default="./data/")
    parse.add_argument("--data_path", type=str, default=None)
    parse.add_argument("--model_load_path", type=str, default=None)
    parse.add_argument("--model_save_path", type=str, default="./result/model/")
    parse.add_argument("--model_save_iter", type=int, default=20)
    parse.add_argument("--log_path", type=str, default="./result/")
    parse.add_argument("--log_iter", type=int, default=10)

    # Other config
    parse.add_argument("--mode", choices=["train", "test", "predict"], default="predict")
    parse.add_argument("--net", choices=["RNN", "RNNPB", "LSTM", "LSTMPB", "GRU", "GRUPB"], default="LSTMPB")

    return parse.parse_args()


if __name__ == "__main__":

    config = get_config()

