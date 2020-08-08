#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse

import numpy as np

def main(args):
    joint_files = glob.glob(os.path.join(args.data_path,"*/*joint*.txt"))
    feature_files = glob.glob(os.path.join(args.data_path,"*/*camera*.txt")) 
    joint_files.sort()
    feature_files.sort()
    assert len(joint_files) == feature_files or len(joint_files) != 0

    f_data, j_data = [], []
    for i, (joint, feature) in enumerate(zip(joint_files, feature_files)):
        print("Combine {} and {}".format(os.path.basename(joint), 
                                         os.path.basename(feature)))
        j_data.append(np.loadtxt(joint, delimiter=","))
        f_data.append(np.loadtxt(feature, delimiter=","))
    f_data, j_data = np.stack(f_data), np.stack(j_data)

    data = np.concatenate([j_data, f_data[:, :j_data.shape[1]]], axis=-1)
    for i, d in enumerate(data):
        file_name = "data_" + os.path.basename(joint_files[i]).split("_")[-1]
        file_path = os.path.join(os.path.dirname(joint_files[i]), file_name)
        print("save as {}".format(file_path))
        np.savetxt(file_path, d, delimiter=",")




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, default="./")

    main(parse.parse_args())


