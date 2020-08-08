#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

import numpy as np
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import rospy
from sensor_msgs.msg import JointState

from subscriber import JointSubscriber



class GenMotion(object):
    GRIPPER_WIDTH_WHEN_CLOSED = -0.01
    GRIPPER_WIDTH_WHEN_OPENED = 0.01

    joint_topic = "/joint_states"
    joint_name = ["joint1", "joint2", "joint3", "joint4"]
    gripper_name = ["gripper"]
    def __init__(self, joint_service_name, gripper_service_name, rate=2):
        self.rate = rate
        self.interval = rospy.Duration(secs=1. / self.rate)

        rospy.wait_for_service(joint_service_name)
        rospy.wait_for_service(gripper_service_name)

        self.joint_srv = rospy.ServiceProxy(joint_service_name, SetJointPosition)
        self.gripper_srv = rospy.ServiceProxy(gripper_service_name, SetJointPosition)
        self.sub = JointSubscriber(self.joint_topic)

        rospy.sleep(1)

    def set_pos(self, target):
        joint_cur_pos = self.sub.get()

        joint_pos = SetJointPositionRequest()
        joint_pos.joint_position.joint_name = self.joint_name
        joint_pos.joint_position.position = target[:-1]
        print (self.joint_name, ": ", joint_cur_pos[:-1], target[:-1])
        for i in range(len(self.joint_name)):
            print("{}: {} --> {}".format(self.joint_name[i], joint_cur_pos[i], target[i]))
        joint_pos.path_time = 1. / self.rate

        gripper_pos = SetJointPositionRequest()
        gripper_pos.joint_position.joint_name = self.gripper_name
        if target[-1] <= self.GRIPPER_WIDTH_WHEN_CLOSED:
            target[-1] = self.GRIPPER_WIDTH_WHEN_CLOSED
        elif target[-1] >= self.GRIPPER_WIDTH_WHEN_OPENED:
            target[-1] = self.GRIPPER_WIDTH_WHEN_OPENED
        gripper_pos.joint_position.position = [target[-1]]
        print("{}: {} --> {}".format(self.gripper_name[0], joint_cur_pos[-1], target[-1]))
        gripper_pos.path_time = 1. / self.rate
        return joint_pos, gripper_pos

    def step(self, target):
        joint_pos, gripper_pos = self.set_pos(target)
        print("joint ", self.joint_srv(joint_pos))
        print("gripper ", self.gripper_srv(gripper_pos))
        print("=" * 10, "send command", "=" * 10)

    def __call__(self, target):
        self.step(target)



if __name__ == "__main__":
    node = rospy.init_node("moton_generator")
    gen = GenMotion('/goal_joint_space_path', 'goal_tool_control', rate=5)
    target_data = np.loadtxt("./denorm_target_predict.txt", dtype=np.float32, delimiter=",")[:, :5]
    print(target_data.shape)
    raw_input("press key to start")
    for i in range(target_data.shape[0]):
        print("Step", i)
        #raw_input("Step")
        gen(target_data[i])
        time.sleep(1. / gen.rate)




        








