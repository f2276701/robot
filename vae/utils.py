#!/usr/bin/env python
#coding:utf-8
'''
Created on ($ date)
Update  on
Author:
Team:
Github: 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""This module includes helper functions"""

import time
import glob
import os

import torch
from torchvision import utils


def load_model(model_path, model_name):
    pt_files = glob.glob(os.path.join(model_path, model_name))
    if not pt_files:
        raise TypeError("No trained model exits!")
    pt_files.sort()
    pt_file = pt_files[-1]
    print("load {}".format(pt_file))
    return pt_file


def make_result_img(tensors, **kwargs):
    nrow = len(tensors)
    result_img = torch.cat(tensors)
    grid = utils.make_grid(result_img, nrow=result_img.shape[0] // nrow, **kwargs)
    return grid


class Logger(object):
    def __init__(self, model):
        self.model = model
        self.total_time = 0
        self.start_time = time.time()

    def call(self, epoch):
        consume_time = time.time() - self.start_time
        self.total_time += consume_time
        self.start_time = time.time()

        self.display_loss(epoch, consume_time)
        if hasattr(self.model, 'writer'):
            self.tensorboard_summary(epoch)

        utils.save_image(self.model.result_img.cpu(),
                         "{}{}_epoch{}_result.png".format(self.model.log_path, 
                                                          self.model.name, 
                                                          epoch))

    def display_loss(self, cur_epoch, time):
        content = []
        for name, value in self.model.loss_list.items():
            content.append("{}: {:6f}".format(name, value))
        content = ", ".join(content)
        print("epoch: {}/{}, {}, time: {:6f}".format(cur_epoch, 
                                                     self.model.epoch, 
                                                     content, 
                                                     time))
        if hasattr(self.model, "vali_loss_list"):
            content = []
            for name, value in self.model.vali_loss_list.items():
                content.append("{}: {:6f}".format(name, value))
            content = ", ".join(content)
            print("     vali loss: {}".format(content)) 

    def tensorboard_summary(self, epoch):
        total_loss = {"train":self.model.loss_list["total_loss"]}
        rec_loss = {"train":self.model.loss_list["rec_loss"]}
        kld_loss = {"train":self.model.loss_list["kld_loss"]}
        if hasattr(self.model, 'vali_loss_list'):
            total_loss.update({"vali":self.model.vali_loss_list["total_loss"]})
            rec_loss.update({"vali":self.model.vali_loss_list["rec_loss"]})
            kld_loss.update({"vali":self.model.vali_loss_list["kld_loss"]})
            
        self.model.writer.add_scalars('Total loss', total_loss, epoch)
        self.model.writer.add_scalars('Bce loss', rec_loss, epoch)
        self.model.writer.add_scalars('Kld loss', kld_loss, epoch)

        #parameter log
        for name, value in self.model.net.named_parameters():
            self.model.writer.add_histogram(name, value, epoch)

        self.model.writer.add_image("result_{:0>6d}".format(epoch), self.model.result_img, 
                                    global_step=epoch)

    def __call__(self, epoch):
        self.call(epoch)


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from torchvision import datasets, transforms 
    transform = transforms.Compose([transforms.Resize([64, 64]),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([.5], [.5])])
    #dataset = CondDataSet(root="./data", train=True,
    #dataset = CondDataSet(root="./data", split="train",
    #print(dataset[0])
    for idx in range(10):
        img = dataset[900]
        #print(img[-1])
        #print(img)
        tmp = [i.view(1,-1,64,64) for i in img[:-1]]
        grid_img = make_result_img(tmp, normalize=True, range=(-1., 1.))
        print(grid_img.shape)
        utils.save_image(grid_img, "./{}.png".format(idx))
    


    






