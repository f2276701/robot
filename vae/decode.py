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

    if args.decode:
        #z = torch.randn(16, args.z_dim)
        z = torch.from_numpy(np.loadtxt("./denorm_output_log.txt", delimiter=",", dtype=np.float32)[:, 5:])
        #for i in range(10):
        for i in range(z.shape[0]):
            #z_ = z.repeat(50, 1)
            #z_[:, i] = torch.linspace(-4, 4, 50)
            logits, result = vae.decoder(z[i, None])
            grid_img = make_result_img([result], normalize=True, range=(-1., 1.))
            utils.save_image(grid_img, "./rec/VAE_decode_result_{:0>6d}.png".format(i))

    if args.rec_img:
        dataset = datasets.ImageFolder(root=args.data_path, transform=transform)
        data, label = [], []
        for i, j in dataset:
            data.append(i)
            label.append(j)
        data = torch.stack(data)
        label = torch.tensor(label)

        _, result, _, _ = vae(data)
        grid_img = make_result_img([data, result],normalize=True, range=(-1., 1.))
        utils.save_image(grid_img, "{}VAE_reconstract_result.png".format(args.log_path))
    

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--z_dim", type=int, default=7)
    parse.add_argument("--data_path", type=str, default="./data/20200706/sp/")
    parse.add_argument("--model_load_path", type=str, default="./result/result200706_z7b2/VAE/model/")
    #parse.add_argument("--log_path", type=str, default="./result/result200605_88/VAE/")
    parse.add_argument("--log_path", type=str, default="./")

    parse.add_argument("--decode", type=bool, default=True)
    parse.add_argument("--rec_img", type=bool, default=False)

    main(parse.parse_args())
