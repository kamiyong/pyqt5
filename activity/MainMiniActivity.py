"""
Created on 2021-1-12 15:11:22
@author: kamiyong
@file: MainActivity
@description: 主窗口类
"""
import sys

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QHBoxLayout
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QColor, QPalette

from view.TextView import TextView
from view.EditText import EditText
from view.ImageView import ImageView

from constant.Context import Key, ctx
from constant import Global

from util.ClickCounter import ClickCounter

from activity.CalibrationActivity import CalibrationActivity


class Pane(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # 宽度是否可变
        self.isImmutableWidth = True
        # 高度是否可变
        self.isImmutableHeight = True

    def setIsImmutable(self, valueWidth, valueHeight):
        self.isImmutableWidth = valueWidth
        self.isImmutableWidth = valueHeight

    def change(self, w, h):
        width = w if self.isImmutableWidth else self.width()
        height = h if self.isImmutableHeight else self.height()
        self.resize(width, height)


class MainWindow(QWidget):
    # 头部的高度
    HEAD_HEIGHT = 30
    # 头部下面的线条高度
    LINE_HEIGHT = 1
    RIGHT_LABEL_WIDTH = 40
    # 默认的背景颜色
    BG_COLOR = "#0C172D"
    # 边界间隔
    PADDING = 10

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # 获取屏幕尺寸相关信息
        self.ag = QApplication.desktop().availableGeometry()
        self.maxWidth = self.ag.width()
        self.maxHeight = self.ag.height()

        self.real = QApplication.desktop().geometry()

        self.windowPreState = None

        self.initWidth = 320
        self.initHeight = self.maxHeight
        # 全局存储当前屏幕参数
        ctx.put(Key.WINDOW_INIT_SIZE, [self.initWidth, self.initHeight])
        ctx.put(Key.SCREEN_AVAIL_SIZE, [self.maxWidth, self.maxHeight])
        ctx.put(Key.SCREEN_REAL_SIZE, [self.real.width(), self.real.height()])
        ctx.put(Key.WINDOW_HEAD_HEIGHT, MainWindow.HEAD_HEIGHT)
        # 先创建实例，让引用一直持有该实例，不至于被回收，导致窗口闪现
        self.ca = CalibrationActivity()

        # 头部标题栏
        self.head = Pane(self)
        # 条码输入框
        self.__barcode = EditText()
        # 流程展示控件
        self.__resultBox = QLabel()
        # 显示最终结果控件
        self.__displayBox = TextView()
        # 摄像头开启控件, 在这里只新建对象，不要将其添加到窗口中
        # 因为图层遮挡，到后面得不到鼠标事件
        self.cameraIcon = ImageView()

        # 当前窗口关闭回调事件
        self.closeWindowCallback = None
        # 关闭点击计数器
        self.counter = ClickCounter()

    def setCloseWindowCallback(self, callback):
        self.closeWindowCallback = callback

    def initStyle(self):
        self.setObjectName("window")
        self.resize(self.initWidth, self.initHeight)
        self.setWindowTitle(Global.project_name)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.move(self.maxWidth - self.initWidth, 0)
        # self.setStyleSheet(style)

    def display(self):
        self.initStyle()
        self.createHead()
        self.createLine()
        self.createBody()
        self.show()

    def createHead(self):
        self.head.resize(self.initWidth, MainWindow.HEAD_HEIGHT)
        self.head.move(0, 0)

        innerLayout = QHBoxLayout()
        innerLayout.layout().setContentsMargins(0, 0, 0, 0)

        self.head.setLayout(innerLayout)

        adjust = TextView("校准")
        adjust.setObjectName("adjust")
        adjust.setFixedSize(MainWindow.RIGHT_LABEL_WIDTH, MainWindow.HEAD_HEIGHT)
        adjust.setAlignment(Qt.AlignCenter)
        adjust.setMousePressCallback(self.clickedEvent)

        # 右边
        # 隐藏窗口控件
        dis = ImageView()
        dis.setFixedSize(MainWindow.RIGHT_LABEL_WIDTH, MainWindow.HEAD_HEIGHT)
        dis.setObjectName("dis")
        dis.setCallback(self.clickedEvent)

        # 最大最小化窗口控件
        maxi = ImageView()
        maxi.setFixedSize(MainWindow.RIGHT_LABEL_WIDTH, MainWindow.HEAD_HEIGHT)
        maxi.setObjectName("maxi")
        maxi.setCallback(self.clickedEvent)

        # 关闭窗口控件
        close = ImageView()
        close.setFixedSize(MainWindow.RIGHT_LABEL_WIDTH, MainWindow.HEAD_HEIGHT)
        close.setObjectName("close")
        close.setCallback(self.clickedEvent)
        close.setDoubleClick(self.doubleClickEvent)

        innerLayout.addStretch(0)

        innerLayout.addWidget(adjust)
        innerLayout.addWidget(dis)
        innerLayout.addWidget(maxi)
        innerLayout.addWidget(close)

    def createLine(self):
        line = QWidget(self)
        line.resize(self.initWidth, MainWindow.LINE_HEIGHT)
        line.move(0, MainWindow.HEAD_HEIGHT)
        line.setObjectName("line")

    def createBody(self):
        body = QWidget(self)
        body.resize(self.initWidth, self.initHeight - MainWindow.HEAD_HEIGHT - MainWindow.LINE_HEIGHT)
        body.move(0, MainWindow.HEAD_HEIGHT + MainWindow.LINE_HEIGHT)

        box_percent = 0.99
        # 左边盒子宽度
        leftBoxWidth = int(self.initWidth * box_percent)

        # 盒子高度
        boxHeight = self.initHeight - MainWindow.HEAD_HEIGHT - MainWindow.LINE_HEIGHT
        # padding
        padding = int(self.initWidth * (1 - box_percent))
        # 条码输入框高度
        barcodeHeight = 30
        # 流程信息展示区高度
        displayBoxHeight = int(boxHeight * 0.7)

        # ############左边区域#################
        leftBox = QWidget(body)
        leftBox.resize(leftBoxWidth, boxHeight)
        leftBox.move(padding, padding)
        leftBox.setObjectName("leftBox")
        # leftBox.setStyleSheet("background-color: #CCECDA;")

        # 内部垂直布局
        leftBoxInnerBox = QVBoxLayout()
        leftBoxInnerBox.setSpacing(10)
        leftBox.setLayout(leftBoxInnerBox)

        # 条码输入框
        self.__barcode.setParent(leftBox)
        self.__barcode.setObjectName("barcode")
        self.__barcode.setMaximumSize(leftBoxWidth, barcodeHeight)

        leftBoxInnerBox.addWidget(self.__barcode)

        # 流程展示区
        self.__displayBox.setParent(leftBox)
        self.__displayBox.resize(leftBoxWidth, displayBoxHeight)
        self.__displayBox.setObjectName("displayBox")
        # 设置该属性，如果文本超过容器的宽度自动换行
        self.__displayBox.setWordWrap(True)
        leftBoxInnerBox.addWidget(self.__displayBox)
        self.__displayBox.setAlignment(Qt.AlignTop)

        # 显示最终结果的地方
        self.__resultBox.setParent(leftBox)
        self.__resultBox.setObjectName("resultBox")
        self.__resultBox.resize(leftBoxWidth, boxHeight - displayBoxHeight - barcodeHeight - 20)
        self.__resultBox.setMaximumSize(leftBoxWidth, boxHeight - displayBoxHeight - barcodeHeight - 20)
        self.__resultBox.setMinimumHeight(boxHeight - displayBoxHeight - barcodeHeight - 20)
        self.__resultBox.setAlignment(Qt.AlignCenter)
        self.__resultBox.setPalette(QPalette(QColor(255, 0, 0, 1)))
        # 设置该属性，如果文本超过容器的宽度自动换行
        self.__resultBox.setWordWrap(True)
        leftBoxInnerBox.addWidget(self.__resultBox)

    def resizeEvent(self, event):
        """
        窗口尺寸变化事件
        :param event: {@link QEvent}
        :return: no return
        """
        # print("MainWindow Size[w=", self.width(), ", h=", self.height(), "]")
        self.head.change(self.width(), MainWindow.HEAD_HEIGHT)

    def getBarcodeEditText(self):
        """
        获取到条码输入框实例
        :return: {@link QEditText}
        """
        return self.__barcode

    def getDisplayBox(self):
        """
        获取流程展示控件
        :return: {@link QLabel}
        """
        return self.__displayBox

    def getResultBox(self):
        """
        获取结果显示控件
        :return: {@link QLabel}
        """
        return self.__resultBox

    def clickedEvent(self, obj: QWidget):
        name = obj.objectName()
        if name == "adjust":  # 校准窗口按钮点击事件
            print("adjust is clicked.")
            # 只有在窗口隐藏的时候才开启
            if self.ca.isHidden():
                self.ca.display()
        elif name == "dis":  # 最小化窗口
            print("dis is clicked.")
            self.setWindowState(Qt.WindowMinimized)
        elif name == "maxi":  # 最大化窗口点击事件
            print("maxi is clicked.")
        elif name == "close":  # 关闭窗口点击事件
            print("single click.")

    def doubleClickEvent(self, obj: QWidget):
        name = obj.objectName()
        if name == "close":
            print("close window")
            if self.closeWindowCallback is not None:
                self.closeWindowCallback()
            self.close()
            sys.exit(0)

    def changeEvent(self, e: QEvent):
        # WindowStateChange
        # print("Window change: {}".format(e.type()))
        pass

