#!/usr/bin/env python
#coding:utf-8

import argparse


def get_config():
    parse = argparse.ArgumentParser()

    # Common config
    parse.add_argument("--img_size", type=list, default=[3, 64, 64])
    parse.add_argument("--log_path", type=str, default="./result/")
    parse.add_argument("--model_load_path", type=str, default="./model/")
    parse.add_argument("--cuda", type=bool, default=True)
    parse.add_argument("--window_size", type=int, default=32)

    # LSTM config
    parse.add_argument("--unit_nums", type=int, default=100)
    parse.add_argument("--in_dims", type=int, default=12)
    parse.add_argument("--out_dims", type=int, default=12)
    parse.add_argument("--closed_index", type=int, default=5)
    parse.add_argument("--closed_rate", type=float, default=1)
    parse.add_argument("--pb_dims", type=int, default=7)
    parse.add_argument("--delay", type=int, default=1)
    parse.add_argument("--config_path", type=str, default="./model/")

    # VAE config
    parse.add_argument("--vae_z_dims", type=int, default=7)

    # CVAE config
    parse.add_argument("--cvae_z_dims", type=int, default=20)
    parse.add_argument("--sample_num", type=int, default=50)
    return parse.parse_args()


