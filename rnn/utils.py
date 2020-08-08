#!/usr/bin/env python
#coding:utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""This module includes helper functions"""

import time
import glob
import os

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils


def add_normal_noise(array):
    noise = np.random.normal(0.0, 0.05, array.shape).astype(np.float32)
    return noise + array


def sampling(data_num, data_step, pattern="cycle", noise=None, **pattern_args):
    def cycle(data_step, x_rad=0.8, y_rad=0.8, x_center=0, y_center=0):
        theta = np.linspace(0, 2 * np.pi, data_step)
        x = x_rad * np.sin(theta) + x_center
        y = y_rad * np.cos(theta) + y_center

        return np.stack([x, y], axis=-1)

    def sin(data_step, amp=0.8, f=1, phase=0):
        step = np.linspace(0, 2 * np.pi * f, data_step)
        x = step/(2 * np.pi * f) * amp * 2 - amp
        y = np.sin(step + phase) * amp

        return np.stack([x, y], axis=-1)

    def cos(data_step, amp=0.8, f=1, phase=0):
        step = np.linspace(0, 2 * np.pi * f, data_step)
        x = step/(2 * np.pi * f) * amp * 2 - amp
        y = np.cos(step + phase) * amp

        return np.stack([x, y], axis=-1)

    def high_freq(data_step):
        x = [-0.5, 0.5] * (data_step // 2)
        return np.array(x).reshape(-1, 1)

    data = []
    for _ in range(data_num):
        sample = eval(pattern)(data_step, **pattern_args)
        data.append(sample)
    data = np.stack(data, axis=0)

    if noise:
        noised_data = add_normal_noise(data)
    else:
        noised_data = data[:]

    return data.astype(np.float32), noised_data.astype(np.float32)


def make_fig(data, plot_3d=False, figsize=(20, 20), **kwargs):
    fig = plt.figure(figsize=figsize, **kwargs)
    fig.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.95, hspace=0.2)
    cmap = plt.get_cmap("tab10")
    for idx, (output, target) in enumerate(data, 1):
        row = (output.shape[-1] if output is not None else target.shape[-1]) - 5 + 2
        ax = fig.add_subplot(row, 1, 1)
        ax.set_ylim(-1, 1)
        ax.set_yticks(np.linspace(-1, 1, 11))
        for j in range(4):
            if output is not None:
                ax.plot(np.arange(1, output.shape[0] + 1), output[:, j], 
                         linestyle="--", color=cmap(j), 
                         label="Predicted joint" + str(j))
            if target is not None:
                ax.plot(np.arange(target.shape[0]), target[:, j], 
                         linestyle="-", color=cmap(j), 
                         label="Actual joint" + str(j))

        ax = fig.add_subplot(row, 1, 2, sharey=ax)
        if output is not None:
            ax.plot(np.arange(1, output.shape[0] + 1), output[:, 4], 
                             linestyle="--", color=cmap(3), 
                             label="Predicted joint" + str(j))
        if target is not None:
            ax.plot(np.arange(target.shape[0]), target[:, 4], 
                             linestyle="-", color=cmap(3), 
                             label="Actual joint" + str(j))

        for j in range(5, row - 2 + 5):
            ax = fig.add_subplot(row, 1, j - 5 + 3, sharey=ax)
            if output is not None:
                ax.plot(np.arange(1, output.shape[0] + 1), output[:, j], 
                        linestyle="--", color=cmap(0), 
                        label="Predicted feature" + str(j))
            if target is not None:
                ax.plot(np.arange(1, target.shape[0] + 1), target[:, j], 
                        linestyle="-", color=cmap(0),
                        label="Actual feature" + str(j))
        ax.set_xlabel("Time step")
    return fig


def load_model(model_path, model_name):
    pt_files = glob.glob(os.path.join(model_path, model_name))
    if not pt_files:
        raise TypeError("No trained model exits!")
    pt_files.sort()
    pt_file = pt_files[-1]
    print("load {}".format(pt_file))
    return pt_file


class Logger(object):
    def __init__(self, model):
        self.model = model
        self.total_time = 0
        self.start_time = time.time()

    def call(self, epoch):
        consume_time = time.time() - self.start_time
        self.total_time += consume_time
        self.start_time = time.time()
        save_name = "{}{}_epoch{}_result.png".format(self.model.log_path, 
                                                     self.model.name, 
                                                     epoch)

        self.display_loss(epoch, consume_time)
        if hasattr(self.model, 'writer'):
            self.tensorboard_summary(epoch)

        if type(self.model.result_img) is Tensor:
            utils.save_image(self.model.result_img, save_name)
        elif type(self.model.result_img) is Figure:
            self.model.result_img.savefig(save_name)

    def display_loss(self, cur_epoch, time):
        def convert_msg(loss_list):
            content = []
            for name, value in loss_list:
                content.append("{}: {:6f}".format(name, value))
            content = ", ".join(content)
            return content

        content = convert_msg(self.model.loss_list.items())
        print("epoch: {}/{}, {}, time: {:6f}".format(cur_epoch, 
                                                     self.model.epoch, 
                                                     content, 
                                                     time))
        if hasattr(self.model, "vali_loss_list"):
            content = convert_msg(self.model.vali_loss_list.items())
            print("*****vali loss: {}".format(content)) 

    def tensorboard_summary(self, epoch):
        for name in self.model.loss_list.keys():
            loss = {"train": self.model.loss_list[name]}
            if hasattr(self.model, 'vali_loss_list'):
                loss.update({"vali":self.model.vali_loss_list[name]})
            self.model.writer.add_scalars(name, loss, epoch)

        for name, value in self.model.rnn.named_parameters():
            self.model.writer.add_histogram(name, value, epoch)

        if type(self.model.result_img) is Tensor:
            self.model.writer.add_image("result", self.model.result_img,
                                        global_step=epoch)
        elif type(self.model.result_img) is Figure:
            self.model.writer.add_figure("result", self.model.result_img, global_step=epoch)

    def __call__(self, epoch):
        self.call(epoch)


class DataGenerator(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.target[index]


def gen_data(batch_size):
    # generate synthetic data
    #sin_data, noised_sin_data = sampling(50, 100, "sin", noise=True)
    #cos_data, noised_cos_data = sampling(50, 100, "cos", phase=0.5*np.pi, noise=True)
    #data = np.concatenate([sin_data[:, :75], cos_data[:, :75]], 0)
    data, noised_data = sampling(100, 100, "high_freq", noise=True)
    #noised_data = np.concatenate([noised_sin_data[:, :75], noised_cos_data[:, :75]], 0)
    data_generator = DataGenerator(noised_data, data)

    pb_list = torch.zeros([data.shape[0], 2])
    pb_list[:50, 1] = 1
    pb_list[50:, 0] = 1

    data_loader = DataLoader(data_generator, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=6)

    data = torch.from_numpy(data)
    return data, data_loader, pb_list


def load_data(data_path):
    assert os.path.exists(data_path), "{} is not exist!".format(data_path)
    data_list = sorted(glob.glob(os.path.join(data_path, "target*.txt")))
    print("load data")
    print(data_list)
    assert len(data_list), "{} is empty!".format(data_path)
    data = np.stack([np.loadtxt(path, dtype=np.float32, delimiter=",") for path in data_list])
    noised_data = add_normal_noise(data)
    data_generator = DataGenerator(noised_data, data)
    data_loader = DataLoader(data_generator, 
                             batch_size=len(data_list), 
                             shuffle=False, 
                             num_workers=6)

    return torch.from_numpy(data), data_loader


if __name__ == "__main__":
    data, _ = load_data("./data/20200605/train/")
    #files = glob.glob("./data/20200605/train/target*0.txt")
    #files.sort()
    #data = []
    #for f in files:
    #    data.append(np.loadtxt(f, delimiter=","))
    #data = np.stack(data)

    fig = make_fig([[data[0], data[0]]])
    plt.show()



