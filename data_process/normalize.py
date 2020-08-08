#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import argparse

import numpy as np

class Normalizer:
    def __init__(self, data, norm_range=[-0.8, 0.8]):
        if isinstance(norm_range, (tuple, list)):
            norm_range = np.array(norm_range)
        if norm_range.ndim == 1:
            norm_range = norm_range.reshape(1, -1)
        if norm_range.shape[0] < data.shape[-1]:
            norm_range = np.repeat(norm_range, data.shape[-1], axis=0)

        data_range = []
        for j in range(data.shape[-1]):
            max_val, min_val = -float("inf"), float("inf")
            for i in range(data.shape[0]):
                max_val = max(max_val, data[i, :, j].max())
                min_val = min(min_val, data[i, :, j].min())
            data_range.append([min_val, max_val])
        data_range = np.stack(data_range)

        self.data_range = data_range
        self.norm_range = norm_range
        self._scaler = (norm_range[:, 1] - norm_range[:, 0]) /\
                      (data_range[:, 1] - data_range[:, 0]) 
        self._min = norm_range[:, 0] - data_range[:, 0] * self._scaler

    def __call__(self, data):
        _scaler, _min = self._scaler, self._min
        while data.ndim > _scaler.ndim:
            _scaler = np.expand_dims(_scaler, 0)
            _min = np.expand_dims(_min, 0)
        return data * _scaler + _min


def main(args):
    data_files = glob.glob(os.path.join(args.data_path,"*/data*.txt"))
    data_files.sort()

    data = []
    for f in data_files:
        data.append(np.loadtxt(f, delimiter=","))
    data = np.stack(data)

    norm = Normalizer(data, norm_range=[-0.8, 0.8])
    normed_data = norm(data)
    pb_list = normed_data[:, -1, 5:]
    for i, d in enumerate(normed_data):
        file_name = "target_" + os.path.basename(data_files[i]).split("_")[-1]
        file_path = os.path.join(os.path.dirname(data_files[i]), file_name)
        print("save as {}".format(file_path))
        np.savetxt(file_path, d, delimiter=",")
    dir_name = os.path.dirname(os.path.dirname(data_files[0]))

    np.savetxt(os.path.join(dir_name, "norm_scale.txt"), norm._scaler, delimiter=",")
    np.savetxt(os.path.join(dir_name, "norm_min.txt"), norm._min, delimiter=",")
    np.savetxt(os.path.join(dir_name, "pb_list.txt"), pb_list, delimiter=",")




if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, default="../data/20200618_2/processed/")

    main(parse.parse_args())





