#!/usr/bin/env python
#coding:utf-8


import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import PIL
from PIL import Image


class GoalWidget(QWidget):
    def __init__(self, goal, parent=None):
        super(GoalWidget, self).__init__(parent)
        self.button1 = QPushButton("", self)
        self.button1.setGeometry(10, 10, 266, 266)
        self.button1.setIconSize(QSize(256, 256))
        self.button1.clicked.connect(self.goal1)

        self.button2 = QPushButton("", self)
        self.button2.setGeometry(10, 276, 266, 266)
        self.button2.setIconSize(QSize(256, 256))
        self.button2.clicked.connect(self.goal2)

        self.label_img = QLabel(self)
        self.label_img.setGeometry(300, 10, 512, 512)
        goal = self.pil2qimg(goal)
        goal = goal.scaled(512, 512, Qt.KeepAspectRatio)
        self.label_img.setPixmap(goal)
        self.label_img_txt = QLabel("Current Goal", self)
        self.label_img_txt.setGeometry(500, 522, 80, 30)

        self.decision = None

    def set_goal(self, goal1, goal2):
        assert type(goal1) == type(goal2)

        if isinstance(goal1, Image.Image):
            goal1, goal2 = self.pil2qimg(goal1), self.pil2qimg(goal2)
            goal1 = goal1.scaled(256, 256, Qt.KeepAspectRatio)
            goal2 = goal2.scaled(256, 256, Qt.KeepAspectRatio)
        elif isinstance(goal1, np.ndarray):
            goal1, goal2 = cv2.resize(goal1, (256, 256)), cv2.resize(goal2, (256, 256))
            goal1 = self.np2qimg(goal1)
            goal2 = self.np2qimg(goal2)
        else:
            raise TypeError("only support Image type or np.ndarray type")

        self.button1.setIcon(QIcon(goal1))
        self.button2.setIcon(QIcon(goal2))

    def np2qimg(self, img):
        h, w, c = img.shape
        bytesPerLine = 3 * w
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        qimage = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap

    def pil2qimg(self, img):
        data = img.tobytes("raw","RGB")
        qimage = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        return pixmap 

    def goal1(self):
        self.decision = 0
        goal = self.button1.icon().pixmap(256, 256)
        goal = goal.scaled(512, 512, Qt.KeepAspectRatio)
        self.label_img.setPixmap(goal)

    def goal2(self):
        self.decision = 1
        goal = self.button2.icon().pixmap(256, 256)
        goal = goal.scaled(512, 512, Qt.KeepAspectRatio)
        self.label_img.setPixmap(goal)
        self.label_img.setPixmap(goal)



if __name__ == "__main__":
    app = QApplication([])
    goal1, goal2 = Image.open("./gen_goal000000.png"), Image.open("./gen_goal000000.png")
    window = GoalWidget(goal1)
    window.set_goal(goal1, goal2)
    window.show()
    app.exec_()




