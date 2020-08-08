#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils

from utils import make_result_img


class DataSet(Dataset):
    """Data loading
    """

    def __init__(self, data, transform=None, target_transform=None):
        self.data = []
        for i in data:
            self.data.extend(i)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        img = target_img = self.loader(self.data[idx][0])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return img, target_img

    def __len__(self):
        return len(self.data)

    def loader(self, path):
        with open(path, 'rb'):
            img = Image.open(path)
            return img.convert('RGB')


class CondDataSet(Dataset):
    """Data loading
    loading data and corresponding same class conditional data 
    """
    def __init__(self, envdata, goaldata, transform=None, target_transform=None, 
                 input_mode="env"):
        """
        Args:
            input_mode: str, either be "env" or "goal". "env" means env img \
                        will be input to the model with goal img as condition, vice versa.
        """

        assert input_mode in ["env", "goal"], "expect input_type in ['env', 'goal'], \
                                               got {}".format(input_type)

        # arrange cond data by class

        self.env_data = []
        for i in envdata:
            self.env_data.extend(i)
        self.goal_data = goaldata
        self.transform = transform
        self.target_transform = target_transform
        self.input_mode = input_mode

    def __getitem__(self, idx):
        env_img, label = self.env_data[idx]
        goal_index = np.random.choice(len(self.goal_data[label]))
        goal_img = self.goal_data[label][goal_index][0]
        env_img, goal_img = self.loader(env_img), self.loader(goal_img)

        if self.input_mode == "env":
            img, cond_img, target_img = env_img, goal_img, env_img
        elif self.input_mode == "goal":
            img, cond_img, target_img = goal_img, env_img, goal_img
        else:
            raise Exception("input mode error: expect be either 'env' or 'goal', \
                             got '{}'".format(self.input_mode))

        if self.transform is not None:
            img = self.transform(img)
            cond_img = self.transform(cond_img)
        if self.target_transform is not None:
            target_img = self.target_transform(target_img)
        return img, cond_img, target_img

    def __len__(self):
        return len(self.env_data)

    def loader(self, path):
        with open(path, 'rb'):
            img = Image.open(path)
            return img.convert('RGB')


def load_data(root, batch_size, transform=None, target_transform=None, 
              input_mode="env", condition=True):
    assert input_mode in ["env", "goal"], "expect input_type in ['env', 'goal'], \
                                           got {}".format(input_type)

    dirs = glob.glob(os.path.join(root, "env/goal*"))
    dirs.sort()
    print(dirs)
    assert len(dirs) != 0, "env data is empty"
    env_data = []
    for i, d in enumerate(dirs):
        data = datasets.ImageFolder(d)
        env_data.append([])
        for sample in data.samples:
            env_data[-1].append((sample[0], i))

    if condition:
        dirs = glob.glob(os.path.join(root, "goal/goal*"))
        dirs.sort()
        assert len(dirs) != 0, "goal data is empty"
        goal_data = []
        for i, d in enumerate(dirs):
            data = datasets.ImageFolder(d)
            goal_data.append([])
            for sample in data.samples:
                goal_data[-1].append((sample[0], i))

        print("=" * 5, "Loading cond data", "=" * 5)
        dataset = CondDataSet(env_data, goal_data,
                              transform=transform,
                              target_transform=target_transform,
                              input_mode=input_mode)
    else:
        print("=" * 5, "Loading data", "=" * 5)
        dataset = DataSet(env_data,
                          transform=transform,
                          target_transform=target_transform)

    img_size = dataset[0][0].shape
    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                             shuffle=True, 
                             num_workers=8)
    return dataset, data_loader


if __name__ == "__main__":
    dataset, data_loader = load_data("./data/20200604/train/", 16, transforms.ToTensor(), transforms.ToTensor())
    x = iter(data_loader).next()
    grid_img = make_result_img(x[:-1])
    utils.save_image(grid_img, "./grid_img.png")

