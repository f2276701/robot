#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import rospy

POS_MAP = {"home_r":[0.529, -0.854, -0.617, 1.9], 
           "home_l":[0, -0.854, -0.617, 1.9], 
           "dumpling":[0.660, -0.620, 0.537, 1.532],
           "hotdog":[1.085, -0.233, 0.239, 1.440],
           "banana":[0.423, -0.282, 0.276, 1.371],
           "donut":[0.792, 0.137, -0.193, 1.477],
           "corn":[0.265, 0.366, -0.476, 1.503],
           "shrimp":[0.568, 0.471, -0.658, 1.431],
           "top_left":[-0.408, 0.765, -1.304, 1.804],
           "top_right":[-0.680, 0.012, -0.195, 1.606],
           "down_right":[-0.187, -0.293, 0.173, 1.531],
           "gripper_on":[0.01],
           "gripper_off":[-0.01]}
JOINT_TIME = 5.
GRIPPER_TIME = 2.

def main():
    rospy.wait_for_service("/goal_joint_space_path")
    rospy.wait_for_service("/goal_tool_control")

    joint_srv = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)
    gripper_srv = rospy.ServiceProxy("/goal_tool_control", SetJointPosition)

    joint_name = ["joint1", "joint2", "joint3", "joint4"]
    gripper_name = ["gripper"]

    while True:
        inputs = raw_input("choose action pattarn: [1/2]")
        if inputs not in ["1", "2"]:
            print("Only accept pattarn 1 or 2")
            continue

        if inputs == "1":
            actions = ["dumpling", "gripper_on", "home_l", "top_right", "gripper_off", "home_l", "home_r", 
                       "banana", "gripper_on", "home_l", "down_right", "gripper_off", "home_l", "home_r",]
        elif inputs == "2":
            actions = ["hotdog", "gripper_on", "home_l", "top_right", "gripper_off", "home_l", "home_r", 
                       "donut", "gripper_on", "home_l", "down_right", "gripper_off", "home_l", "home_r",]


        for action in actions:
            target = POS_MAP[action]
            name = joint_name if "gripper" not in action else gripper_name
            time = JOINT_TIME if "gripper" not in action else GRIPPER_TIME
            srv = joint_srv if "gripper" not in action else gripper_srv
            pos = set_pos(target, name, time)
            print(action, srv(pos))
            rospy.sleep(time)

def set_pos(target, name, time):
    pos = SetJointPositionRequest()
    pos.joint_position.joint_name = name
    pos.joint_position.position = target
    pos.path_time = time
    return pos


if __name__ == "__main__":
    main()
