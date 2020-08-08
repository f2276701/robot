#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob 
import os

import numpy as np

"""This scripts extract orig trajectory into \
   multiple independent trajectory"""

def main(args):
    data_files = glob.glob(os.path.join(args.data_path,"*/target_0*.txt"))
    data_files.sort()
    assert len(data_files), "no data found in {}".format(args.data_path)

    data = []
    for f in data_files:
        data.append(np.loadtxt(f, delimiter=","))
    data = np.stack(data)

    #avoiding diffrent sequence length after shifting
    if data.shape[1] % args.num != 0:
        data = np.concatenate([data, data[:, -(args.num - data.shape[1] % args.num):, :]], 1)  

    for i, data_file in enumerate(data_files):
        output_dir = os.path.dirname(data_file)
        for j in range(args.num):
            output_file = os.path.join(output_dir, "target_5hz_{:0>6d}.txt".format(i * args.num + j)) 
            np.savetxt(output_file, data[i, j::args.num, :], delimiter=",")

    pb_list_fpath = os.path.join(args.data_path, "pb_list.txt")
    assert os.path.exists(pb_list_fpath), "no pb file found in {}".format(args.data_path)
    pb_list = np.loadtxt(pb_list_fpath, delimiter=",")
    pb_list = pb_list.repeat(args.num, axis=0)
    np.savetxt(os.path.join(args.data_path, "pb_list_5hz.txt"), pb_list, delimiter=",")



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, default="../data/20200705/processed/")
    parse.add_argument("--num", type=int, default=2)
    main(parse.parse_args())



