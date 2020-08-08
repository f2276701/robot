#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""This module includes helper functions"""

import argparse
import glob
import os
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

from rnn.utils import make_fig

def main(args):
    data_files = glob.glob(os.path.join(args.data_path,"*/target_0*.txt"))
    data_files.sort()

    data = []
    for f in data_files:
        data.append(np.loadtxt(f, delimiter=","))
    data = np.stack(data)

    for i, d in enumerate(data):
        fig = make_fig([[None, d]])
        fig.savefig(data_files[i].replace(".txt", ".png"))



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, default="../data/20200706_z7/processed/")

    main(parse.parse_args())
