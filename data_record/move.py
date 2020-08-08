#!/usr/bin/env python
#coding:utf-8

'''
Created on ($ date)
Update  on
Author:
Team:
Github: 
'''

import time
import rospy
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest

pos_map = {"home_r":[0.529, -0.854, -0.617, 1.9], 
           "home_l":[0, -0.854, -0.617, 1.9], 
           "dumpling":[0.660, -0.620, 0.537, 1.532],
           "hotdog":[1.085, -0.233, 0.239, 1.440],
           "banana":[0.423, -0.282, 0.276, 1.371],
           "donut":[0.792, 0.137, -0.193, 1.477],
           "corn":[0.265, 0.366, -0.476, 1.503],
           "shrimp":[0.568, 0.471, -0.658, 1.431],
           "top_left":[-0.408, 0.765, -1.304, 1.804],
           "top_right":[-0.680, 0.012, -0.195, 1.606],
           "down_right":[-0.187, -0.293, 0.173, 1.531]}
TIME = 5.

def main():
    rospy.wait_for_service("/goal_joint_space_path")
    rospy.wait_for_service("/goal_tool_control")
    joint_srv = rospy.ServiceProxy("/goal_joint_space_path", SetJointPosition)

    print(pos_map.keys())
    while True:
        pos = raw_input()
        target = pos_map[pos]
            
        pos = set_pos(target)
        print(joint_srv(pos))
        time.sleep(TIME)

def set_pos(target):
    joint_name = ["joint1", "joint2", "joint3", "joint4"]
    joint_pos = SetJointPositionRequest()
    joint_pos.joint_position.joint_name = joint_name
    joint_pos.joint_position.position = target
    joint_pos.path_time = TIME
    return joint_pos


if __name__ == "__main__":
    main()
