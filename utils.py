#!/usr/bin/env python
#coding:utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import glob
import os

import torch


def load_model(model_path, model_name):
    pt_files = glob.glob(os.path.join(model_path, model_name))
    if not pt_files:
        raise TypeError("No trained model exits!")
    pt_files.sort()
    pt_file = pt_files[-1]
    print("load {}".format(pt_file))
    return pt_file


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

    def __len__(self):
        return len(self.deque)
