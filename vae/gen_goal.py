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

import torch
from torchvision import utils, datasets, transforms

from vae import CVAE
#from vae_inception import CVAE
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
    pt_file = load_model(args.model_load_path, "*10.pt")
    cvae.load_state_dict(torch.load(pt_file))
    cvae.eval()

    if args.decode:
        #z_data = [torch.randn(data.shape[0], args.z_dim), data]
        #_, rec_img = cvae.decoder(*z_data)
        #grid_img = make_result_img([data, rec_img], normalize=True, range=(-1., 1.))
        #utils.save_image(grid_img, "{}CVAE_gen_result.png".format(args.log_path))

        cond_img = data[None, 0].repeat([32, 1, 1, 1])
        z_data = [torch.randn(cond_img.shape[0], args.z_dim), cond_img]
        _, rec_img = cvae.decoder(*z_data)
        #grid_img = make_result_img([rec_img], normalize=True, range=(-1., 1.))
        #utils.save_image(grid_img, "{}CVAE_gen_result_same_cond.png".format(args.log_path))
        for i in range(rec_img.shape[0]):
            utils.save_image(rec_img[i], "{}goal{:0>6d}.png".format(args.log_path, i), normalize=True, range=(-1., 1.))

    if args.gen_seq:
        for i, d in enumerate(data):
            cond_img = data[None, i]
            z_data = [torch.randn(1, args.z_dim), cond_img]
            _, rec_img = cvae.decoder(*z_data)
            grid_img = make_result_img([cond_img, rec_img], normalize=True, range=(-1., 1.))
            utils.save_image(grid_img, "{}res_state-goal/CVAE_gen_{:0>6d}.png".format(args.log_path, i))
            utils.save_image(rec_img, "{}res_goal/CVAE_gen_{:0>6d}.png".format(args.log_path, i),
                             normalize=True, range=(-1., 1.))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--z_dim", type=int, default=20)
    parse.add_argument("--data_path", type=str, default="./data/sp/")
    parse.add_argument("--model_load_path", type=str, default="./result/result200706_z7b2/fine-tuning_CVAE/model/")
    parse.add_argument("--log_path", type=str, default="./result/result200706_z7b2/fine-tuning_CVAE/")

    parse.add_argument("--decode", type=bool, default=False)
    parse.add_argument("--gen_seq", type=bool, default=True)

    main(parse.parse_args())
