#!/usr/bin/env python
#coding:utf-8

'''
Created on
Update  on
Author: JumpLion
Team:
Github: https://f2276701.github.io/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import sys
import argparse
sys.path.append("./")

import numpy as np
#from utils import Normalizer


def main(args):
    #data_files = glob.glob(os.path.join(args.data_path,"*predict*.txt"))
    #data_files.sort()

    #data = []
    #for f in data_files:
    #    data.append(np.loadtxt(f, delimiter=","))
    #data = np.stack(data)
    data = np.loadtxt("./pb_log.txt", delimiter=",")[None]
    scaler = np.loadtxt("../norm_scale.txt", delimiter=",")
    min_ = np.loadtxt("../norm_min.txt", delimiter=",")
    normed_data = (data - min_.reshape(1, 1, -1)[:, :, 5:]) / scaler.reshape(1, 1, -1)[:, :, 5:]
    np.savetxt("./denorm_pb_log.txt", normed_data[0], delimiter=",")
    input()

    for i, d in enumerate(normed_data):
        file_name = "denorm_target_" + os.path.basename(data_files[i]).split("_")[-1]
        file_path = os.path.join(os.path.dirname(data_files[i]), file_name)
        print("save as {}".format(file_path))
        np.savetxt(file_path, d, delimiter=",")
    dir_name = os.path.dirname(os.path.dirname(data_files[0]))
    #np.savetxt(os.path.join(dir_name, "normed_data_range.txt"), denorm.data_range, delimiter=",")
    #np.savetxt(os.path.join(dir_name, "denorm_range.txt"), denorm.norm_range, delimiter=",")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, default="../")

    main(parse.parse_args())





