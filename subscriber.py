#!/usr/bin/env python
#coding:utf-8


import os
from threading import Lock

import cv2
import rospy
import sensor_msgs.msg as msg
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import pandas as pd

__all__ = ['ImageRecorder', 'JointRecorder', 'Recorder']

class Buffer(object):
    def __init__(self):
        self._data = None
        self._is_empty = True
        self.lock = Lock()

    def push(self, data):
        with self.lock:
            self._data = data

    def pop(self):
        with self.lock:
            data = self._data
            self._data = None
        return data

    @property
    def is_empty(self):
        with self.lock:
            return self._data is None


class BasicSubscriber(object):
    msg_type = None

    def __init__(self, topic, sync=False):
        self._start_time = rospy.get_time()
        self.data_buffer = Buffer()
        self.topic = topic

        if sync:
            self.sub = message_filters.Subscriber(topic, self.msg_type)
        else:
            self.sub = rospy.Subscriber(self.topic, self.msg_type, self.callback)

    def get(self):
        raise NotImplementedError()

    def callback(self, data):
        self.data_buffer.push(data)


class ImageSubscriber(BasicSubscriber):
    code = "bgr8"
    msg_type = msg.Image

    def __init__(self, *args, **kwargs):
        super(ImageSubscriber, self).__init__(*args, **kwargs)
        self.bridge = CvBridge()

    def get(self):
        if not self.data_buffer.is_empty:
            img = self.data_buffer.pop()
            try:
                cv_img = self.bridge.imgmsg_to_cv2(img, self.code)
            except CvBridgeError as e:
                print(e)
            return cv_img


class JointSubscriber(BasicSubscriber):
    msg_type = msg.JointState
    joint_names = ["joint1", "joint2", "joint3", "joint4", "gripper"]

    def __init__(self, *args, **kwargs):
        super(JointSubscriber, self).__init__(*args, **kwargs)

    def get(self):
        if not self.data_buffer.is_empty:
            joint_state = self.data_buffer.pop()
            return [self.get_joint_position(joint_state, name) for name in self.joint_names]

    def get_joint_position(self, joint_state, joint_name):
        return joint_state.position[joint_state.name.index(joint_name)]



if __name__ == "__main__":
    import time
    rospy.init_node("test")
    sub = ImageSubscriber("/camera/color/image_raw")
    #rospy.sleep(1)
    while not rospy.is_shutdown():
        print(sub.get())
        #rospy.sleep(1)
        time.sleep(0.5)
        

