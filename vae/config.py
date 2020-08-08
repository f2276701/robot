#!/usr/bin/env python
#coding:utf-8


import argparse
import os


def get_config():
    parse = argparse.ArgumentParser()

    # Moel config
    parse.add_argument("--net", choices=["VAE", "CVAE"], default="CVAE")
    parse.add_argument("--z_dim", type=int, default=20)
    parse.add_argument("--prior_dist", choices=["G", "B"], default="G", 
                        help="G means Gaussian distribution, \
                              B means iBernoulli distribution")

    # Training config
    parse.add_argument("--beta", type=int, default=1)
    parse.add_argument("--batch_size", type=int, default=16)
    parse.add_argument("--lr", type=float, default=1e-4)
    parse.add_argument("--epoch", type=int, default=100)
    parse.add_argument("--cuda", type=bool, default=True)
    parse.add_argument("--validate", type=bool, default=False)
    parse.add_argument("--tensorboard", type=bool, default=True)

    parse.add_argument("--input_mode", choices=["env", "goal"], default="env")
    parse.add_argument("--data_path", type=str, default="./data/")
    parse.add_argument("--model_load_path", type=str, default=None)
    parse.add_argument("--model_save_path", type=str, default="./result/model/")
    parse.add_argument("--model_save_iter", type=int, default=10)
    parse.add_argument("--log_path", type=str, default="./result/")
    parse.add_argument("--log_iter", type=int, default=5)
    parse.add_argument("--rec_num", type=int, default=16)

    # Other config
    parse.add_argument("--mode", choices=["train", "test"], default="train")
    parse.add_argument("--seed", type=int, default=123)
    return parse.parse_args()

if __name__ == "__main__":
    config = get_config()

