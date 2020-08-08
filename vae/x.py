#!/usr/bin/env python
#coding:utf-8

'''
Created on ($ date)
Update  on
Author:
Team:
Github: 
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse

import numpy as np
import torch
from torchvision import utils, datasets, transforms

from vae import VAE
from utils import make_result_img, load_model


def main(args):
    transform = transforms.Compose([transforms.Resize([64, 64], 1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([.5], [.5])])

    vae = VAE(img_size=[3, 64, 64], z_dim=args.z_dim)
    pt_file = load_model(args.model_load_path, "*.pt")
    vae.load_state_dict(torch.load(pt_file))
    vae.eval()

    z = np.loadtxt("../denorm_pb_list.txt", delimiter=",", dtype=np.float32)
    x = np.linspace(z[0], z[5], 100)
    print(x.shape)
    _, result = vae.decoder(torch.from_numpy(x))
    grid_img = make_result_img([result], normalize=True, range=(-1., 1.))
    utils.save_image(grid_img, "./x.png")



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--z_dim", type=int, default=7)
    #parse.add_argument("--data_path", type=str, default="./data/20200701/sp/1/")
    parse.add_argument("--model_load_path", type=str, default="./result/result200705_z7b2/VAE/model/")
    #parse.add_argument("--log_path", type=str, default="./result/result200605_88/VAE/")
    parse.add_argument("--log_path", type=str, default="./")


    main(parse.parse_args())
