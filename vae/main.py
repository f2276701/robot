#!/usr/bin/env python
#coding:utf-8


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import copy
import os

import torch
from torchvision import transforms, utils
import numpy as np

from vae import VAE, CVAE
#from vae_inception import VAE, CVAE
from model import Model
from config import get_config
from utils import make_result_img, load_model
from load_data import load_data


def main(args):
    #########################################
    ###          Loading Dataset          ###
    #########################################
    if args.prior_dist == "G":
        transform = transforms.Compose([transforms.Resize([64, 64], 1),
                                        transforms.ToTensor(),
                                        transforms.Normalize([.5], [.5])])
    elif args.prior_dist == "B":
        transform = transforms.Compose([trainsforms.Resize([64, 64], 1),
                                        transforms.ToTensor()])
    target_transform = copy.deepcopy(transform)
    transform.transforms.insert(0, transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
    transform.transforms.insert(0, transforms.ColorJitter(0.4, 0.4, 0.4))
                                
    print(transform.transforms)

                                        
    train_root = os.path.join(args.data_path, "train")
    train_data, train_loader = load_data(train_root, args.batch_size,
                                         transform, target_transform,
                                         input_mode=args.input_mode,
                                         condition=True if args.net == "CVAE" else False)
    test_root = os.path.join(args.data_path, "test")
    _, test_loader = load_data(test_root, args.batch_size,
                               transform, target_transform,
                               input_mode=args.input_mode,
                               condition=True if args.net == "CVAE" else False)
    if args.validate:
        vali_root = os.path.join(args.data_path, "validation")
        _, vali_loader = load_data(vali_root, args.batch_size,
                                   transform, target_transform,
                                   input_mode=args.input_mode,
                                   condition=True if args.net == "CVAE" else False)
    else:
        vali_loader = None
    img_size=train_data[0][0].shape

    #########################################
    ###          Building Model           ###
    #########################################
    model = Model(args)
    model.build(eval(args.net), img_size=img_size, z_dim=args.z_dim) 
    if args.model_load_path:
        model_name = "*.pt"
        pt_file = load_model(args.model_load_path, model_name)
        model.load(pt_file)

    if args.mode == "train":
        model.train(train_loader, vali_loader)
    elif args.mode == "test":
        result_img = model.test(test_loader)
        result = data + [result_img]
        if args.prior_dist == "G":
            grid_img = make_result_img(result, normalize=True, range=(-1., 1.))
        else:
            grid_img = make_result_img(result)
        utils.save_image(grid_img, "{}{}_test.png".format(args.log_path, args.net))


if __name__ == "__main__":
    args = get_config()
    if args.seed > -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    main(args)

