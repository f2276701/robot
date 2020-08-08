#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import os

import pandas as pd
from PIL import Image


joint_names = ["joint1", "joint2", "joint3", "joint4", "gripper"]
img_fname = lambda s: "camera_image_{:0>6d}.png".format(s)

def align(joint_list, img_list):
    max_idx, max_len = 0, 0
    for i in range(len(joint_list)):
        if joint_list[i].shape[0] > max_len:
            max_len = joint_list[i].shape[0]
            max_idx = i
    print("aligned length", max_len)

    for idx in range(len(joint_list)):
        if idx == max_idx:
            continue
        _, joint_list[idx] = joint_list[max_idx].align(joint_list[idx], axis=0, method="pad")

        fill_img = img_list[idx][-1]
        while len(img_list[idx]) < max_len:
            img_list[idx].append(fill_img)
        assert joint_list[idx].shape[0] == len(img_list[idx]) == max_len
    return joint_list, img_list


def main(args):
    joint_files = sorted(glob.glob(os.path.join(args.joint_path, "*/joint*.txt")))
    print("\n".join(joint_files))
    img_files = sorted(glob.glob(os.path.join(args.img_path, "*/*image*")))
    print("\n".join(img_files))
    assert len(joint_files) == len(img_files) != 0

    joint_list, img_list = [], []
    for joint_f, img_f in zip(joint_files, img_files):
        joint_list.append(pd.read_csv(joint_f, names=joint_names))
        img_list.append(sorted(glob.glob(os.path.join(img_f, "camera*.png"))))
        assert joint_list[-1].shape[0] == len(img_list[-1]) != 0, \
               "Joint sequence and corresponding image sequence must have same sequence length"

    joint_list, img_list = align(joint_list, img_list)
    for idx, (joint_df, img_paths) in enumerate(zip(joint_list, img_list)):
        joint_fname = os.path.basename(joint_files[idx])
        output_joint_dir = os.path.join(args.output_path, 
                                         os.path.dirname(joint_files[idx]).split("/")[-1]) 
        if not os.path.exists(output_joint_dir):
            os.makedirs(output_joint_dir)
        output_joint_path = os.path.join(output_joint_dir, joint_fname)
        joint_df.to_csv(output_joint_path, index=False, header=False)

        img_dir = os.path.join(args.output_path, "/".join(img_list[idx][0].split("/")[-3:-1]))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        for img_no, img_path in enumerate(img_paths):
            img = Image.open(img_path)
            img_path = os.path.join(img_dir, img_fname(img_no))
            img.save(img_path, quality=95)



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--joint_path", type=str, default="../data/20200618_2/orig/")
    parse.add_argument("--img_path", type=str, default="../data/20200618_2/orig/")
    parse.add_argument("--output_path", type=str, default="./data/20200618_2/processed/")

    main(parse.parse_args())

            

