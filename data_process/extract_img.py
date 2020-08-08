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
import glob
import os
import sys
sys.path.append("../")

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from vae import VAE
from vae.utils import load_model


def main(args):
    transform = transforms.Compose([transforms.Resize([64, 64], 1),
                                    transforms.ToTensor(),
                                    transforms.Normalize([.5], [.5])])

    vae = VAE(img_size=[3, 64, 64], z_dim=args.z_dim)
    pt_file = load_model(args.model_load_path, "*.pt")
    vae.load_state_dict(torch.load(pt_file))
    vae.eval()

    dirs = glob.glob(os.path.join(args.data_path, "goal*/camera*"))
    dirs = [d for d in dirs if os.path.isdir(d)]
    dirs.sort()

    for d in dirs:
        data = []
        files = glob.glob(os.path.join(d, "*.png"))
        files.sort()
        print(d.split("/")[-1])
        for f in files:
            img = Image.open(f)
            img = transform(img)
            data.append(vae.reparam(*vae.encoder(img[None, :, :, :])).detach())
        data = torch.cat(data).cpu().numpy()
        print(d + ".txt")
        np.savetxt(d + ".txt", data, delimiter=",")


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--z_dim", type=int, default=5)
    parse.add_argument("--data_path", type=str, default="../data/20200627/processed/")
    parse.add_argument("--model_load_path", type=str, default="../model/result200627_vae_z5b2/")

    main(parse.parse_args())

