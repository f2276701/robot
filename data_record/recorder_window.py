#!/usr/bin/env python
#coding:utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append("..")

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import PIL
from PIL import Image, ImageQt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import rospy
from sensor_msgs.msg import Image
from sensor_msgs import msg
from subscriber import ImageSubscriber, JointSubscriber



class RecorderWidget(QWidget):
    def __init__(self, rate=10, node_name='recorder', 
                 parent=None):
        super(RecorderWidget, self).__init__(parent)

        self.root_dir = '../data/'
        self.img_filename = 'camera_image'
        self.joint_filename = 'joint_value'
        self._step = 0
        self._index = 0
        self.recording_rate = rate

        self.timer = {'img': QTimer(self), 'joint': QTimer(self)}
        self.timer['img'].timeout.connect(self.img_record)
        self.timer['joint'].timeout.connect(self.joint_record)

        self.startTimer(1000. / rate)
        rospy.init_node(node_name)

        self.img_sub = ImageSubscriber("/camera/color/image_raw")
        rospy.Subscriber("/camera/color/image_raw", msg.Image, self.camera_view)
        self.bridge = CvBridge()

        self.joint_sub = JointSubscriber("/joint_states")

        self.init_ui(parent)
        self.crop_size = ((36, 36 + 260), (250, 250 + 260))
        self.output_img_size= (64, 64)

    def stop_timer(self):
        for t in self.timer.values():
            if t.isActive():
                t.stop()
        self.button_img_record.setText("Img Record\nStart")
        self.button_joint_record.setText("Joint Record\nStart")
        self.button_all_record.setText("All Record\nStart")

    def set_img_name(self):
        self.img_filename = str(self.img_filename_text.text())

    def set_joint_name(self):
        self.joint_filename = str(self.joint_filename_text.text())

    def set_index(self):
        self._index = int(str(self.index_text.text()))

    def select_dir(self):
        dir_name = QFileDialog.getExistingDirectory()
        self.data_dir_text.setText(dir_name)
        self.root_dir = str(dir_name)

    def record_done(self):
        self.stop_timer()
        if hasattr(self, 'joint_file'):
            self.joint_file.close()
            del self.joint_file
        if hasattr(self, "img_path"):
            del self.img_path
        self._step = 0
        self._index += 1
        self.index_text.setText(str(self._index))

    def img_record(self):
        img_name = self.img_filename + '_{:0>6d}.png'.format(self._step)

        img = self.img_sub.get()
        if img is None:
            raise ValueError("the image buffer is empty!")
        else:
            if hasattr(self, "crop_size"):
                img = img[self.crop_size[0][0]:self.crop_size[0][1], self.crop_size[1][0]:self.crop_size[1][1], :]  
            cv2.imwrite(os.path.join(self.img_path, img_name), img)
            self._step += 1
            self.img_record_text.setText(img_name)

    def img_event(self):
        if not hasattr(self, "img_path"):
            self.img_path = os.path.join(self.root_dir, 
                                         self.img_filename + '_{:0>6d}'.format(self._index))
            if not os.path.isdir(self.img_path):
                os.mkdir(self.img_path)

        if not self.timer['img'].isActive():
            self.stop_timer()
            self.timer['img'].start(1000. / self.recording_rate)
            self.button_img_record.setText("Img Record\nStop")
        else:
            self.timer['img'].stop()
            self.button_img_record.setText("Img Record\nStart")

    def joint_record(self):
        joint_val = self.joint_sub.get()
        if joint_val is None:
            raise ValueError("the joint buffer is empty!")
        else:
            self.joint_file.write(",".join([str(value) for value in joint_val]))
            self.joint_file.write("\n")

    def joint_event(self):
        if not hasattr(self, 'joint_file'):
            joint_f_name = os.path.join(self.root_dir, 
                                        self.joint_filename + '_{:0>6d}.txt'.format(self._index))
            self.joint_file = open(joint_f_name, "w")
            self.joint_record_text.setText(joint_f_name)

        if not self.timer['joint'].isActive():
            self.stop_timer()
            self.timer['joint'].start(1000. / self.recording_rate)
            self.button_img_record.setText("Joint Record\nStop")
        else:
            self.timer['joint'].stop()
            self.button_img_record.setText("Joint Record\nStart")

    def all_event(self):
        if not hasattr(self, 'joint_file'):
            joint_f_name = os.path.join(self.root_dir, 
                                        self.joint_filename + '_{:0>6d}.txt'.format(self._index))
            self.joint_file = open(joint_f_name, "w")
            self.joint_record_text.setText(joint_f_name)
        if not hasattr(self, "img_path"):
            self.img_path = os.path.join(self.root_dir, 
                                         self.img_filename + '_{:0>6d}'.format(self._index))
            if not os.path.isdir(self.img_path):
                os.mkdir(self.img_path)

        if not self.timer['joint'].isActive() and not self.timer['img'].isActive():
            self.stop_timer()
            self.timer['joint'].start(1000. / self.recording_rate)
            self.timer['img'].start(1000. / self.recording_rate)
            self.button_all_record.setText("All Record\nStop")
        elif self.timer['joint'].isActive() and self.timer['img'].isActive():
            self.stop_timer()
            self.button_all_record.setText("All Record\nStart")
        else:
            raise Exception("either joint or image is running")

    def camera_view(self, img):
        cv_img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        h, w, c = cv_img.shape
        bytesPerLine = 3 * w
        cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB, cv_img)
        qimage = QImage(cv_img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        if hasattr(self, "crop_size"):
            pixmap = pixmap.copy(QRect(self.crop_size[1][0], self.crop_size[0][0], 260, 260))
        pixmap = pixmap.scaled(260, 260, Qt.KeepAspectRatio)
        self.label_img.setPixmap(pixmap)


    def init_ui(self, parent):
        self.label_img = QLabel(self)
        self.label_img.setGeometry(620, 0, 320, 320)

        self.label_img_record = QLabel("Img Record Log:", self)
        self.label_img_record.setGeometry(0, 0, 200, 30)
        self.img_record_text = QLineEdit(self)
        self.img_record_text.setGeometry(200, 0, 300, 30)
        self.img_record_text.setReadOnly(True)

        self.label_joint_record = QLabel("Joint Record Log:", self)
        self.label_joint_record.setGeometry(0, 35, 200, 30)
        self.joint_record_text = QLineEdit(self)
        self.joint_record_text.setGeometry(200, 35, 300, 30)
        self.joint_record_text.setReadOnly(True)

        self.label_recording_rate = QLabel("Recording Rate", self)
        self.label_recording_rate.setGeometry(0, 70, 200, 30)
        self.recording_rate_text = QLineEdit(self)
        self.recording_rate_text.setGeometry(200, 70, 30, 30)
        self.recording_rate_text.setText(str(self.recording_rate))
        self.recording_rate_text.setReadOnly(True)

        self.label_data_dir = QLabel("Data Dir:", self)
        self.label_data_dir.setGeometry(0, 140, 200, 30)
        self.data_dir_text = QLineEdit(self)
        self.data_dir_text.setGeometry(200, 140, 300, 30)
        self.data_dir_text.setText(os.path.abspath(self.root_dir))
        self.data_dir_text.setReadOnly(True)
        self.button_select_file = QPushButton("Select File", self)
        self.button_select_file.setGeometry(505, 140, 90, 30)
        self.button_select_file.clicked.connect(self.select_dir)

        self.label_joint_filename = QLabel("Joint File Name:", self)
        self.label_joint_filename .setGeometry(0, 175, 200, 30)
        self.joint_filename_text = QLineEdit(self)
        self.joint_filename_text.setGeometry(200, 175, 300, 30)
        self.joint_filename_text.setText(self.joint_filename)
        self.joint_filename_text.returnPressed.connect(self.set_joint_name)
        self.label_index = QLabel("Index:", self)
        self.label_index.setGeometry(515, 175, 50, 30)
        self.index_text = QLineEdit(self)
        self.index_text.setGeometry(565, 175, 30, 30)
        self.index_text.setText(str(self._index))
        self.index_text.returnPressed.connect(self.set_index)

        self.label_img_filename = QLabel("Img File Name", self)
        self.label_img_filename .setGeometry(0, 210, 200, 30)
        self.img_filename_text = QLineEdit(self)
        self.img_filename_text.setText(self.img_filename)
        self.img_filename_text.setGeometry(200, 210, 300, 30)
        self.img_filename_text.returnPressed.connect(self.set_img_name)

        self.label_img_topic = QLabel("Img Topic", self)
        self.label_img_topic.setGeometry(0, 245, 200, 30)
        self.img_topic_text = QLineEdit(self)
        self.img_topic_text.setGeometry(200, 245, 300, 30)
        self.img_topic_text.setText(self.img_sub.topic)
        self.img_topic_text.setReadOnly(True)

        self.label_joint_topic = QLabel("Joint Topic", self)
        self.label_joint_topic.setGeometry(0, 280, 200, 30)
        self.joint_topic_text = QLineEdit(self)
        self.joint_topic_text.setGeometry(200, 280, 300, 30)
        self.joint_topic_text.setText(self.joint_sub.topic)
        self.joint_topic_text.setReadOnly(True)

        self.button_img_record = QPushButton("Img Record\nStart", self)
        self.button_img_record.setGeometry(30, 340, 80, 60)
        self.button_img_record.clicked.connect(self.img_event)

        self.button_joint_record = QPushButton("Joint Record\nStart", self)
        self.button_joint_record.setGeometry(130, 340, 80, 60)
        self.button_joint_record.clicked.connect(self.joint_event)

        self.button_all_record = QPushButton("All Record\nStart", self)
        self.button_all_record.setGeometry(230, 340, 80, 60)
        self.button_all_record.clicked.connect(self.all_event)

        self.button_record_done = QPushButton("Record\nDone", self)
        self.button_record_done.setGeometry(380, 340, 80, 60)
        self.button_record_done.clicked.connect(self.record_done)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.bottom_dock = QDockWidget("Bottom Dock Widget", self)
        self.bottom_dock.setWidget(RecorderWidget(parent=self))
        self.addDockWidget(Qt.BottomDockWidgetArea, self.bottom_dock)

        self.setWindowTitle('armctl_view')


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    #window.setFixedSize(1442, 362 * 2 + 50);
    window.setFixedSize(1442, 460);
    window.show()
    app.exec_()

