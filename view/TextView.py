"""
Created on 2021-1-13 14:47:26
@author: kamiyong
@file: TextView
@description:
"""
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QResizeEvent, QMouseEvent
from PIL import ImageFont


class TextView(QLabel):
    """
        重写QLabel实现文本追加功能
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # 尺寸变化事件
        self.sizeCallback = None
        # 是否文本显示自适应
        self.adjustTextSize = False
        # 鼠标按下事件回调
        self.mousePressCallback = None

    def setCallback(self, callback):
        self.sizeCallback = callback

    def setMousePressCallback(self, callback):
        self.mousePressCallback = callback

    def resizeEvent(self, a0: QResizeEvent) -> None:
        if self.sizeCallback is not None:
            self.sizeCallback(self)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if self.mousePressCallback is not None:
            self.mousePressCallback(self)

    def append(self, text, nextLine=True):
        """
        追加文本
        :param text: 需要追加的文本
        :param nextLine: 是否换行
        :return: no return
        """
        t = self.text()
        if t is None:
            self.setText(text)
        else:
            if nextLine:
                self.setText("{}\n{}".format(t, text))
            else:
                self.setText("{}{}".format(t, text))

    def measureText(self, text):
        font = self.font()
        fontSize = font.pixelSize()
        if fontSize == -1:
            fontSize = font.pointSize()
            if fontSize == -1:
                fontSize = font.pointSizeF()
        imageFont = ImageFont.truetype(r"C:\Windows\Fonts\msyhl.ttc", fontSize)
        # 获取当前文字的宽度和高度
        fontWidth, fontHeight = imageFont.getsize(text)

        return fontWidth, fontHeight

    def fontSize(self):
        font = self.font()
        # 如果使用pixelSize()获取的字体大小为-1 ，则表明设置字体大小使用的是setPointSize()方法
        # 那么就是使用pointSize()回去字体大小, 下一步同理
        fontSize = font.pixelSize()
        if fontSize == -1:
            fontSize = font.pointSize()
            if fontSize == -1:
                fontSize = font.pointSizeF()
        return fontSize