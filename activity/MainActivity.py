"""
Created on 2021-1-12 15:11:22
@author: kamiyong
@file: MainActivity
@description: 主窗口类
"""

from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QHBoxLayout
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QPalette

from view.TextView import TextView
from view.EditText import EditText
from view.ImageView import ImageView

from constant.Context import Key, ctx
from constant import Global


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
    HEAD_HEIGHT = 40
    # 头部下面的线条高度
    LINE_HEIGHT = 1
    RIGHT_LABEL_WIDTH = 60
    # 默认的背景颜色
    BG_COLOR = "#0C172D"
    # 摄像头图片容器的尺寸
    CAMERA_ICON_SIZE = QSize(50, 50)
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

        self.initWidth = 1280
        self.initHeight = 760
        # 全局存储当前屏幕参数
        ctx.put(Key.WINDOW_INIT_SIZE, [self.initWidth, self.initHeight])
        ctx.put(Key.SCREEN_AVAIL_SIZE, [self.maxWidth, self.maxHeight])
        ctx.put(Key.SCREEN_REAL_SIZE, [self.real.width(), self.real.height()])

        # 头部标题栏
        self.head = Pane(self)
        # 条码输入框
        self.barcode = EditText()
        # 流程展示控件
        self.resultBox = QLabel()
        # 显示最终结果控件
        self.displayBox = TextView()
        # 摄像头开启控件, 在这里只新建对象，不要将其添加到窗口中
        # 因为图层遮挡，到后面得不到鼠标事件
        self.cameraIcon = ImageView()

    def initStyle(self):
        self.setObjectName("window")
        self.resize(self.initWidth, self.initHeight)
        self.setWindowTitle(Global.project_name)
        self.setWindowFlags(Qt.FramelessWindowHint)
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

        # 间隔
        leftSpace = QLabel("")
        leftSpace.setFixedSize(0, 10)
        innerLayout.addWidget(leftSpace)

        # 左边图标
        icon = QLabel()
        icon.setFixedSize(25, 25)
        icon.setObjectName("icon")

        # 项目名称
        name = QLabel("AI")
        name.setObjectName("projectName")

        # 版本号
        version = QLabel("v1.2021.01.09")
        version.setObjectName("version")

        innerLayout.addWidget(icon)
        innerLayout.addWidget(name)
        innerLayout.addWidget(version)
        innerLayout.addStretch(1)

        innerLayout.setSpacing(10)
        innerLayout.setAlignment(Qt.AlignHCenter)

        self.head.setLayout(innerLayout)

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

        innerLayout.addStretch(0)

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
        # body.setStyleSheet("background-color: #DA81F5;")

        # 左边盒子宽度
        leftBoxWidth = 400

        # 盒子高度
        boxHeight = 700
        # padding
        padding = 10
        # 条码输入框高度
        barcodeHeight = 30
        # 流程信息展示区高度
        displayBoxHeight = 450

        bannerHeight = 100

        # 右边区域宽度 = 窗口宽度 - 左边盒子宽度 - 左边盒子距离窗口左边的间隔 - 右边盒子和左边盒子的间隔 - 右边和距离窗口右边的间隔
        rightBoxWidth = self.initWidth - leftBoxWidth - padding - padding - padding

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
        self.barcode.setParent(leftBox)
        self.barcode.setObjectName("barcode")
        self.barcode.setMaximumSize(leftBoxWidth, barcodeHeight)
        # self.barcode.setCallback(self.on_edit_change)
        leftBoxInnerBox.addWidget(self.barcode)

        # 流程展示区
        self.displayBox.setParent(leftBox)
        self.displayBox.resize(leftBoxWidth, displayBoxHeight)
        # self.displayBox.setMaximumHeight(displayBoxHeight)
        # self.displayBox.setMinimumHeight(displayBoxHeight)
        self.displayBox.setObjectName("displayBox")
        leftBoxInnerBox.addWidget(self.displayBox)
        self.displayBox.setAlignment(Qt.AlignTop)
        # effect = QGraphicsDropShadowEffect(self)
        # effect.setBlurRadius(12)
        # effect.setOffset(5, -5)
        # effect.setColor(Qt.red)
        # displayBox.setGraphicsEffect(effect)
        self.displayBox.setText("1.获取到条码.\n2.开始识别...\n3.识别结果: OK.")
        self.displayBox.append("4.结束！")
        for i in range(30):
            self.displayBox.append("{}.动作！".format(i))

        # 显示最终结果的地方
        self.resultBox.setParent(leftBox)
        self.resultBox.setObjectName("resultBox")
        self.resultBox.resize(leftBoxWidth, boxHeight - displayBoxHeight - barcodeHeight - 20)
        self.resultBox.setMaximumSize(leftBoxWidth, boxHeight - displayBoxHeight - barcodeHeight - 20)
        self.resultBox.setMinimumHeight(boxHeight - displayBoxHeight - barcodeHeight - 20)
        self.resultBox.setText("OK")
        self.resultBox.setAlignment(Qt.AlignCenter)
        self.resultBox.setPalette(QPalette(QColor(255, 0, 0, 1)))
        leftBoxInnerBox.addWidget(self.resultBox)

        # ############右边区域#################
        rightBox = QWidget(body)
        rightBox.resize(rightBoxWidth, boxHeight)
        rightBox.move(leftBoxWidth + padding + padding, padding)
        rightBox.setObjectName("rightBox")
        # rightBox.setStyleSheet("background-color: #E5E9BF;")

        # 左边区域内部垂直布局
        rightInnerBox = QVBoxLayout()
        rightInnerBox.setAlignment(Qt.AlignCenter)
        rightInnerBox.setSpacing(padding)
        rightBox.setLayout(rightInnerBox)

        # 横幅
        banner = QLabel("Kamiyong Welcome You!")
        banner.setObjectName("banner")
        banner.setMaximumSize(rightBoxWidth - 20, bannerHeight)
        banner.setMinimumSize(rightBoxWidth - 20, bannerHeight)
        rightInnerBox.addWidget(banner)

        image = QLabel()
        image.setObjectName("image")
        image.setMinimumSize(rightBoxWidth - 20, boxHeight - bannerHeight - padding - padding - padding)
        rightInnerBox.addWidget(image)

        # 摄像头图标, 在这里添加到窗口中
        # 这样改控件就在所有的图层最上面，就不会被其他控件遮挡而导致事件失效的问题
        self.cameraIcon.setParent(self)
        self.cameraIcon.setCallback(self.clickedEvent)
        self.cameraIcon.setObjectName("cameraIcon")
        self.cameraIcon.setFixedSize(MainWindow.CAMERA_ICON_SIZE.width(), MainWindow.CAMERA_ICON_SIZE.height())
        self.cameraIcon.move(
            self.initWidth - MainWindow.CAMERA_ICON_SIZE.width() - MainWindow.PADDING,
            MainWindow.HEAD_HEIGHT + MainWindow.LINE_HEIGHT + MainWindow.PADDING
        )

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
        return self.barcode

    def getDisplayBox(self):
        """
        获取流程展示控件
        :return: {@link QLabel}
        """
        return self.displayBox

    def getResultBox(self):
        """
        获取结果显示控件
        :return: {@link QLabel}
        """
        return self.resultBox

    def clickedEvent(self, obj: QWidget):
        name = obj.objectName()
        if name == "dis":  # 最小化窗口
            print("dis is clicked.")
            self.setWindowState(Qt.WindowMinimized)
        elif name == "maxi":  # 最大化窗口点击事件
            print("maxi is clicked.")
        elif name == "close":  # 关闭窗口点击事件
            print("close is clicked.")
            self.close()
        elif name == "cameraIcon":
            print("cameraIcon is clicked.")



