#!/usr/bin/env python
#coding:utf-8

'''
Created on
Update  on
Author: JumpLion
Team:
Github: https://f2276701.github.io/
'''


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import glob

import torch
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from vae import CVAE
from utils import make_result_img, load_model



def main(args):
    transform = transforms.Compose([transforms.Resize([64, 64], 1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([.5], [.5])])
    dataset = datasets.ImageFolder(root=args.data_path, transform=transform)

    data = []
    for i, j in dataset:
        data.append(i)
    data = torch.stack(data)

    cvae = CVAE(img_size=data.shape[1:], z_dim=args.z_dim)
    cvae.eval()
    pt_files = glob.glob(os.path.join(args.model_load_path, "*.pt"))
    pt_files.sort()
    #data_1 = data[0][None, :, :, :].repeat([20, 1, 1, 1])
    #data_2 = data[1][None, :, :, :].repeat([20, 1, 1, 1])

    for i in range(len(pt_files)):
        print(pt_files[i])
        cvae.load_state_dict(torch.load(pt_files[i]))

        #z_data = [torch.randn(data.shape[0], args.z_dim), data]
        z_data = [torch.randn(32, args.z_dim), data[None, 0].repeat([32, 1, 1, 1])]
        _, rec_img = cvae.decoder(*z_data)
        grid_img = make_result_img([data[None, 0].repeat([32, 1, 1, 1]), rec_img], normalize=True, range=(-1., 1.))
        utils.save_image(grid_img, "{}CVAE_gen_result_{:0>2d}.png".format(args.log_path, i))

        #z_data = [torch.randn(data_1.shape[0], args.z_dim), data_1]
        #_, rec_img = cvae.decoder(*z_data)
        #grid_img = make_result_img([data_1, rec_img], normalize=True, range=(-1., 1.))
        #utils.save_image(grid_img, "{}CVAE_1_gen_result_{:0>2d}.png".format(args.log_path, i))

        #z_data = [torch.randn(data_2.shape[0], args.z_dim), data_2]
        #_, rec_img = cvae.decoder(*z_data)
        #grid_img = make_result_img([data_2, rec_img], normalize=True, range=(-1., 1.))
        #utils.save_image(grid_img, "{}CVAE_2_gen_result_{:0>2d}.png".format(args.log_path, i))

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--z_dim", type=int, default=20)
    #parse.add_argument("--data_path", type=str, default="./data/test/1/")
    parse.add_argument("--data_path", type=str, default="./data/20200706/sp/")
    parse.add_argument("--model_load_path", type=str, default="./result/result200706_z7b2/fine-tuning_CVAE/model/")
    parse.add_argument("--log_path", type=str, default="./result/result200706_z7b2/fine-tuning_CVAE/")


    main(parse.parse_args())
