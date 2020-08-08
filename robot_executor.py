#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import time

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import PIL
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import rospy
from sensor_msgs import msg
import torch

from config import get_config
from generator import GenMotion
from nn_model import NNModel
from subscriber import JointSubscriber, ImageSubscriber

from goal_window import GoalWidget
from rnn.utils import make_fig



class Robot:
    img_topic = "/camera/color/image_raw"
    joint_topic = "/joint_states"
    joint_service = "/goal_joint_space_path"
    gripper_service = "/goal_tool_control"
    node_name = "robot_executor"
    joint_name = ['joint1', 'joint2', 'joint3', 'joint4','gripper']

    def __init__(self, args, rate=1):
        self.node = rospy.init_node(self.node_name)
        self.joint_sub = JointSubscriber(self.joint_topic)
        self.img_sub = ImageSubscriber(self.img_topic)
        self.rate = rate

        self.gen_motion = GenMotion(self.joint_service, self.gripper_service, rate=rate) 
        self.model = NNModel(args)

        self.app = QApplication([])
        self.window = GoalWidget(self.model.goal_img)

        self.crop_size = ((36, 36 + 260), (250, 250 + 260))

    def online(self):
        output_log = []
        denorm_log = []
        state_log = []

        cur_joint, state = self.gen_motion.sub.get(), None
        raw_input("press key to start online prediction")
        for i in range(550):
            print("Step:", i)
            if i == 30:
                self.goal_change(cur_img)
            cur_img = self.img_sub.get()
            cur_img = cur_img[self.crop_size[0][0]:self.crop_size[0][1], self.crop_size[1][0]:self.crop_size[1][1], :]  
            output, state, denorm_output = self.model.on_predict(cur_joint, cur_img, state)
            self.gen_motion(denorm_output[0, :5].numpy().tolist())
            state_log.append(np.concatenate([s[0].detach().cpu().numpy() for s in state], -1))

            output_log.append(output[0].numpy())
            denorm_log.append(denorm_output[0].numpy())
            cur_joint = denorm_output[0, :5]
            time.sleep(1. / self.rate)

        output_log = np.stack(output_log)
        denorm_log = np.stack(denorm_log)
        state_log = np.stack(state_log)
        return output_log, state_log, denorm_log
        
    def offline(self, target_data):
        output_log = []
        denorm_log = []
        state_log = []

        #cur_joint, state = self.gen_motion.sub.get(), None
        cur_joint, state = target_data[0][:5], None
        raw_input("press key to start offline prediction")
        for i, target in enumerate(target_data):
            print("Step:", i)
            if i == 80:
                self.model.pem()
            img_feature = target[5:]
            output, state, denorm_output = self.model.off_predict(cur_joint, img_feature, state)
            self.gen_motion(denorm_output[0, :5].numpy().tolist())
            state_log.append(np.concatenate([s[0].detach().cpu().numpy() for s in state], -1))

            output_log.append(output[0].numpy())
            denorm_log.append(denorm_output[0].numpy())
            cur_joint = output[0, :5].tolist()
            time.sleep(1. / self.rate)

        output_log = np.stack(output_log)
        denorm_log = np.stack(denorm_log)
        state_log = np.stack(state_log)
        return output_log, state_log, denorm_log
        
    def goal_change(self, cur_img):
        print("===== goal changing =====")
        prev = self.model.goal
        goal_list, pb_list = self.model.gen_goal(cur_img[:, :, ::-1])
        goal_idx = self.window.set_goal(goal_list[0], goal_list[1])
        self.window.show()
        self.app.exec_()
        self.model.goal = pb_list[self.window.decision]
        print("{} -> {}".format(prev.numpy(), self.model.goal.numpy()))


if __name__ == "__main__":

    args = get_config()
    arm = Robot(args=args, rate=10)
    target_data = []
    target_data.append(np.loadtxt("./data/20200706_z7/processed/goal000/target_000000.txt", dtype=np.float32, delimiter=","))
    target_data.append(np.loadtxt("./data/20200706_z7/processed/goal001/target_000010.txt", dtype=np.float32, delimiter=","))
    target_data.append(np.loadtxt("./data/20200706_z7/processed/goal002/target_000020.txt", dtype=np.float32, delimiter=","))
    target_data.append(np.loadtxt("./data/20200706_z7/processed/goal003/target_000030.txt", dtype=np.float32, delimiter=","))

    output_log, state_log, denorm_log = arm.online()
    #output_log, state_log, denorm_log = arm.offline(target_data[3].tolist())

    np.savetxt("./result/output_log.txt", output_log, delimiter=",")
    np.savetxt("./result/denorm_output_log.txt", denorm_log, delimiter=",")
    np.savetxt("./result/state_log.txt", state_log, delimiter=",")










