"""
@author kamiyong
@date 2021-1-20 10:10:57
@description 校準窗口
"""
import os
import sys
import random

from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QHBoxLayout, QApplication
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QPaintEvent

from util.RectUtil import RectUtil
from util.Resource import res
from view.ImageView import ImageView
from view.TextView import TextView

from constant.Context import ctx, Key


class CalibrationActivity(QMainWindow):
    __IMAGE_WIDTH = 40

    def __init__(self):
        super().__init__()
        rectUtil = RectUtil()
        rectUtil.setFilePath(os.getcwd() + "/resource/config/rect.txt")
        self.rectList = rectUtil.getRectList()
        print(self.rectList[0].toString())
        print(self.rectList[1].toString())
        # self.headHeight = 30
        self.availSize = ctx.get(Key.SCREEN_AVAIL_SIZE)
        self.windowSize = ctx.get(Key.WINDOW_INIT_SIZE)
        self.headHeight = ctx.get(Key.WINDOW_HEAD_HEIGHT)
        print("self.availSize",  self.availSize)
        print("self.windowSize",  self.windowSize)
        print("self.headHeight",  self.headHeight)
        # 校准窗口的宽度 = 屏幕窗口的宽度 - 主窗口的宽度
        self.w = self.availSize[0] - self.windowSize[0]
        self.h = self.availSize[1]
        print("self.w", self.w)
        print("self.h", self.h)
        self.pen = QPen()
        self.pen.setWidth(3)

    def display(self):
        self.createHead()
        # self.createFace()
        css = res.readCssAsStream("cali.qss")
        self.setStyleSheet(css)
        self.resize(self.w, self.h)
        self.setWindowTitle("校准")
        self.move(0, 0)
        self.setWindowOpacity(0.5)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.show()

    def createHead(self):
        head = QWidget(self)
        head.setObjectName("head")
        head.resize(self.w, self.headHeight)
        head.move(0, 0)
        head.setContentsMargins(0, 0, 0, 0)

        headInnerBox = QHBoxLayout()
        headInnerBox.setContentsMargins(0, 0, 0, 0)
        head.setLayout(headInnerBox)

        leftSpace = QLabel()
        leftSpace.resize(10, self.headHeight)
        headInnerBox.addWidget(leftSpace)

        title = TextView()
        title.setObjectName("title")
        title.setText("校准框（请将机台对准框）")
        headInnerBox.addWidget(title)

        # 隐藏窗口控件
        dis = ImageView()
        dis.setFixedSize(CalibrationActivity.__IMAGE_WIDTH, self.headHeight)
        dis.setObjectName("dis")
        dis.setCallback(self.clickedEvent)

        # 关闭窗口控件
        close = ImageView()
        close.setFixedSize(CalibrationActivity.__IMAGE_WIDTH, self.headHeight)
        close.setObjectName("close")
        close.setCallback(self.clickedEvent)

        headInnerBox.addStretch(0)
        headInnerBox.addWidget(dis)
        headInnerBox.addWidget(close)

    def createFace(self, painter):

        for rect in self.rectList:
            pos = rect.position()
            r = QRect(pos[0], (pos[1] + self.headHeight), rect.width(), rect.height() + self.headHeight)
            self.pen.setColor(QColor(rdm(), rdm(), rdm()))
            painter.setPen(self.pen)
            painter.drawRect(r)

    def paintEvent(self, a0: QPaintEvent) -> None:
        print("start paint")
        painter = QPainter()
        painter.begin(self)
        self.createFace(painter)
        painter.end()

    def clickedEvent(self, view: QWidget):
        name = view.objectName()
        if name == "dis":
            self.setWindowState(Qt.WindowMinimized)
        elif name == "close":
            self.close()
            # sys.exit(0)


def randomColor():
    r = int(random.random() * 255 + 1)
    g = int(random.random() * 255 + 1)
    b = int(random.random() * 255 + 1)
    return [r, g, b]


def rdm():
    return int(random.random() * 255 + 1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ca = CalibrationActivity()
    ca.display()
    sys.exit(app.exec_())
