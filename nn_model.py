#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
from torchvision import transforms, utils

from rnn import LSTMPB
from vae import VAE, CVAE
from utils import load_model, HistoryWindow


class NNModel(object):
    def __init__(self, args):
        self.log_path = args.log_path
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.img_size = args.img_size
        self.sample_num = args.sample_num
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize([64, 64], 1),
                                             transforms.ToTensor(),
                                             transforms.Normalize([.5], [.5])])
        self.pil_transform = transforms.ToPILImage(mode="RGB")

        self.norm_scale = np.loadtxt(os.path.join(args.config_path, "norm_scale.txt"),
                                     dtype=np.float32, delimiter=",")[None]
        self.norm_min = np.loadtxt(os.path.join(args.config_path, "norm_min.txt"),
                                   dtype=np.float32, delimiter=",")[None]
        self.pb_list = torch.from_numpy(np.loadtxt(os.path.join(args.config_path, "pb_list.txt"),
                                                   dtype=np.float32, delimiter=","))

        self.kmeans = KMeans(n_clusters=2)
        self.kmeans.fit(self.pb_list)

        print("=" * 5, "Init LSTMPB", "=" * 5)
        self.rnn = LSTMPB(args, pb_unit=self.pb_list[5][None])
        pt_file = load_model(args.model_load_path, "*/*LSTMPB*.pt")
        self.rnn.load_state_dict(torch.load(pt_file))

        print("=" * 5, "Init VAE", "=" * 5)
        self.vae = VAE(img_size=args.img_size, z_dim=args.vae_z_dims)
        pt_file = load_model(args.model_load_path, "*/VAE*.pt")
        self.vae.load_state_dict(torch.load(pt_file))
        self.vae.eval()

        print("=" * 5, "Init CVAE", "=" * 5)
        self.cvae = CVAE(img_size=args.img_size, z_dim=args.cvae_z_dims)
        pt_file = load_model(args.model_load_path, "*/*CVAE*.pt")
        self.cvae.load_state_dict(torch.load(pt_file))
        self.cvae.eval()

        self.norm_mode = {"joint":[0, 1, 2, 3, 4],
                          "visual":[5, 6, 7, 8, 9, 10, 11]}
        self.norm_mode["all"] = self.norm_mode["joint"] + self.norm_mode["visual"]

        self.global_step = 0
        self.his_log = HistoryWindow(maxlen=args.window_size)

        #visualize current goal
        _, goal = self.vae.decoder(self.denorm(self.goal, "visual"))
        goal = ((goal[0] * .5 + .5) * 255).to(torch.int8)
        self.goal_img = self.pil_transform(goal)

    def on_predict(self, cur_joint, cur_img, state=None):
        cur_joint = torch.Tensor(cur_joint)[None]
        cur_img = self.transform(cur_img[:, :, ::-1])[None]
        utils.save_image(cur_img[0], "./result/visual_{:0>6d}.png".format(self.global_step), 
                         normalize=True, range=(-1, 1))

        img_feature = self.vae.reparam(*self.vae.encoder(cur_img))
        inputs = torch.cat([cur_joint, img_feature], axis=-1).detach()
        inputs = self.norm(inputs).to(torch.float32)

        outputs, state = self.rnn.step(inputs, state)
        outputs, state = outputs.detach().cpu(), \
                         (state[0].detach().cpu(), state[1].detach().cpu())
        self.global_step += 1
        return outputs, state, self.denorm(outputs).to(torch.float32)

    def off_predict(self, cur_joint, img_feature, state=None):
        assert isinstance(cur_joint, (list, np.ndarray))
        assert isinstance(img_feature, (list, np.ndarray))

        cur_joint = torch.Tensor(cur_joint).to(torch.float32)[None]
        img_feature = torch.Tensor(img_feature).to(torch.float32)[None]

        inputs = torch.cat([cur_joint, img_feature], axis=-1)
        outputs, state = self.rnn.step(inputs, state)
        outputs, state = outputs.detach().cpu(), \
                         (state[0].detach().cpu(), state[1].detach().cpu())

        self.his_log.put([outputs, inputs, state])
        return outputs, state, self.denorm(outputs).to(torch.float32)
        
    def gen_goal(self, visual_img):
        visual_img = self.transform(visual_img)[None].repeat(self.sample_num, 1, 1, 1)
        sampled_z = torch.randn(self.sample_num, self.cvae.z_dim)
        _, gen_goals = self.cvae.decoder(z=sampled_z, cond=visual_img)
        pb_list = self.vae.reparam(*self.vae.encoder(gen_goals)).detach().cpu()
        #for i in range(gen_goals.shape[0]):
        #    utils.save_image(gen_goals[i], "{}gen_goal{:0>6d}.png".format("./", i),normalize=True, range=(-1., 1.))

        pb_label = self.kmeans.predict(pb_list.numpy())
        print(pb_label)
        pb_list = torch.stack([pb_list[pb_label==0].mean(0), 
                               pb_list[pb_label==1].mean(0)])
        _, goal_list = self.vae.decoder(pb_list)
        pb_list = self.norm(pb_list, "visual")
        goal_list = ((goal_list * .5 + .5) * 255).to(torch.int8)
        goal_list = [self.pil_transform(goal) for goal in goal_list]
        return goal_list, pb_list

    def pem(self):
        assert len(self.his_log), "the history window is empty!"
        for param in self.rnn.parameters():
            param.requires_grad = False
        self.rnn.pb_unit = nn.Parameter(self.rnn.pb_unit, requires_grad=True)
        optim_param = [param for param in self.rnn.parameters() if param.requires_grad == True]
        optim = torch.optim.Adam(optim_param, lr=0.01)
        mse_loss = nn.MSELoss()

        pred_his, actual_his, state_his = self.his_log.get()
        pb_log = []
        for i in range(80):
            log = []
            cur_input = torch.cat([pred_his[:, 0, :5], actual_his[:, 0, 5:]], dim=-1)
            state = state_his[0]
            for step in range(1, len(state_his)):
                cur_input, state = self.rnn.step(cur_input, state)
                log.append(cur_input)
            log = torch.stack(log, dim=1)
            loss = mse_loss(log[0, :, 5:], actual_his[0, 1:, 5:]) + \
                   (self.rnn.pb_unit - self.pb_list).pow(2).mean()
            pb_log.append(self.rnn.pb_unit.data.clone())
            loss.backward()
            optim.step()
            print("PEM loss, step {}, loss: {}".format(i, loss.item()))

    @property
    def goal(self):
        return self.rnn.pb_unit

    @goal.setter
    def goal(self, pb):
        if pb.ndim == 1:
            pb = torch.unsqueeze(pb, 0)
        self.rnn.pb_unit = pb

    def norm(self, inputs, mode="all"):
        assert mode in ["joint", "visual", "all"]
        i_slice = self.norm_mode[mode]
        return inputs * self.norm_scale[:, i_slice] + self.norm_min[:, i_slice]

    def denorm(self, outputs, mode="all"):
        assert mode in ["joint", "visual", "all"]
        i_slice = self.norm_mode[mode]
        return (outputs - self.norm_min[:, i_slice]) / self.norm_scale[:, i_slice]


if __name__ == "__main__":
    from config import get_config
    import glob
    import numpy as np
    from PIL import Image
    import cv2
    from torchvision import utils
    from vae.utils import make_result_img
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from goal_window import GoalWidget
    import matplotlib.pyplot as plt

    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    from PyQt5.QtWidgets import *

    args = get_config()
    model = NNModel(args)
    #model.predict(np.zeros([1, 8]), Image.open("./camera_image_000000.png"))
    #joint, img = np.zeros([8], dtype=np.float32), np.zeros([3, 64, 64], dtype=np.uint8)
    #model.predict(joint, img)
    #model.goal = np.ones([1, 3])
    img = cv2.imread("./data/20200706_z7/orig/goal001/camera_image_000010/camera_image_000080.png")
    goal_list, pb_list = model.gen_goal(img[:, :, ::-1])
    goal_list = model.gen_goal(img[:, :, ::-1])
    for i in range(goal_list.shape[0]):
        utils.save_image(goal_list[i], "{}gen_goal{:0>6d}.png".format("./", i),normalize=True, range=(-1., 1.))
    #utils.save_image(goal_list[0], "{}gen_goal1.png".format("./"),normalize=True, range=(-1., 1.))
    #utils.save_image(goal_list[1], "{}gen_goal2.png".format("./"),normalize=True, range=(-1., 1.))

    app = QApplication([])
    init_goal = model.rnn.pb_unit
    init_goal = model.denorm(init_goal, "visual")
    _, init_goal = model.vae.decoder(init_goal)
    init_goal =  ((init_goal * .5 + .5) * 255).to(torch.int8)
    init_goal = model.pil_transform(init_goal[0].detach().cpu())
    window = GoalWidget(init_goal)
    window.set_goal(goal_list[0], goal_list[1])
    window.show()
    app.exec_()
    print("got", window.decision)


    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1, projection="3d")
    #pb = pb_list.numpy()
    #pb = model.norm(pb, "visual")
    #print(pb.shape, pb.mean(0)[None].shape)
    #input()
    #train_pb = np.loadtxt("./pb_list.txt", delimiter=",")
    #pb = np.concatenate([train_pb, pb], 0)
    #for i in range(len(pb)):
    #    ax.scatter(pb[i, 0], pb[i, 1], pb[i, 2], label=str(i))
    #ax.legend()
    #plt.show()
    #img_files = glob.glob("./data/20200627/processed/goal000/camera_image_000000/*.png")
    #img_files.sort()

    #log = []
    #for i in range(len(img_files)):
    #    img = cv2.imread(img_files[i])
    #    img = model.transform(img[:, :, ::-1])[None]
    #    img_feature = model.vae.reparam(*model.vae.encoder(img))

    #    log.append(img_feature[0])
    #log = torch.stack(log, 0).detach().cpu().numpy()
    ##log = np.loadtxt("./data_process/1.txt", delimiter=",")
    #log = model.norm(log, "visual")
    ##target = np.loadtxt("./target_5hz_000000.txt", dtype=np.float32, delimiter=",")
    #target = np.loadtxt("./data/20200627/processed/goal000/target_000000.txt", dtype=np.float32, delimiter=",")

    #cmap = plt.get_cmap("tab10")
    #plt.plot(log[:, 3], linestyle="-")
    #plt.plot(target[:, 8], linestyle="--")
    #plt.show()


















