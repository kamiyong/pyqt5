"""
@author kamiyong
@date 2021-1-16 09:41:35
@description 文本区域
"""
from PyQt5.QtWidgets import QWidget, QScrollBar, QLabel
from PyQt5.QtCore import Qt

from view.TextView import TextView


class TextArea(QWidget):
    """
    自定义文本展示区域（PS：不能键盘输入那种）
    功能：能自动拉长文本区域并且带有滑动条
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)
        # 滚动条的宽度
        # 对于水平方向的滚动条其实是高度
        self.scrollBarWidth = 15

        # 文本区
        self.textContainer = TextView(self)
        self.textContainer.setContentsMargins(5, 0, 0, 0)
        # 设置居顶部显示
        self.textContainer.setAlignment(Qt.AlignTop)
        # 设置文本区域根据文本进行自适应
        self.textContainer.adjustSize()
        self.textContainer.setCallback(self.sizeChange)

        # 垂直滚动条
        self.vBar = QScrollBar(Qt.Vertical, self)
        # self.vBar.sliderMoved.connect(self.dragV)
        self.vBar.valueChanged.connect(self.changeVertical)
        # 一开始隐藏
        self.vBar.setVisible(False)
        # 在水平方向上文字的总长度是否超过容器的宽度标志位
        # 只要出现一次超过，那么 self.overHorizontal就永远是True
        # 除非将超过容器宽度的文本都去除
        self.overHorizontal = False
        # 水平方向上滑块上一次的值
        self.hPreValue = 0

        # 水平滚动条
        self.hBar = QScrollBar(Qt.Horizontal, self)
        # self.hBar.sliderMoved.connect(self.dragH)
        self.hBar.valueChanged.connect(self.changeHorizontal)
        # 初始化的时候隐藏自身
        self.hBar.setVisible(False)

        # 在垂直方向上文字的总高度是否超过容器的高度标志位
        # 只要出现一次超过，那么 self.overVertical就永远是True
        # 除非将超过容器高度的文本都去除
        self.overVertical = False
        # 垂直方向上滑块上一次的值
        self.vPreValue = 0

    def setSize(self, width, height):
        """
        设置TextArea的宽度和高度
        :param width:  宽度
        :param height: 高度
        :return: no return
        """
        self.resize(width, height)
        self.setMaximumSize(width, height)
        self.setMinimumSize(width, height)

        # 设置垂直方向ScrollBar的宽度和高度
        # 垂直方向ScrollBar的高度等于TextArea的高度
        self.vBar.resize(self.scrollBarWidth, height - self.scrollBarWidth)
        # 移动到指定位置
        self.vBar.move(width - self.scrollBarWidth, 0)

        # 设置水平方向的ScrollBar的宽度和高度
        # 水平方向ScrollBar的高度为self.scrollBarWidth
        self.hBar.resize(width, self.scrollBarWidth)
        self.hBar.move(0, height - self.scrollBarWidth)

        # 设置文本区域的高度和宽度
        self.textContainer.resize(width, height)
        self.setStyleSheet("background-color: #088A68;")
        self.textContainer.setStyleSheet("border: 1px solid #FF0000; color: #FFFFFF;")

    def setMaximumSize(self, maxw: int, maxh: int) -> None:
        super().setMaximumSize(maxw, maxh)

    def setClassName(self, textAreaName=None, labelName=None, vBarName=None, hBarName=None):
        """
        设置ObjectName
        :param textAreaName: TextArea的ObjectName
        :param labelName: 文本区域的ObjectName
        :param vBarName: 垂直ScrollBar的ObjectName
        :param hBarName: 水平ScrollBar的ObjectName
        :return:
        """
        self.setObjectName(textAreaName)
        self.textContainer.setObjectName(labelName)
        self.hBar.setObjectName(hBarName)
        self.vBar.setObjectName(vBarName)

    def append(self, text):
        self.textContainer.adjustSize()
        self.textContainer.append(text)

        margin = self.contentsMargins()
        # 如果文本长度超过了容器的宽度， 将标准位置为True
        # 并且显示水平滑动条
        if self.textContainer.width() > (self.width() - margin.left() - margin.right()):
            self.overHorizontal = True
            self.hBar.setVisible(True)

        # 如果文本总高度度超过了容器的宽度， 将标准位置为True
        # 并且显示垂直滑动条
        if self.textContainer.height() > (self.height() - margin.top() - margin.bottom()):
            self.overVertical = True
            self.vBar.setVisible(True)

    def dragHortizontal(self, value):
        """
        滑块水平方向上的拖拽事件
        :argument value
        :return:
        """
        print("H: " + str(value))

    def dragVertical(self, value):
        """
        滑块垂直方向上的拖拽事件
        :argument value
        :return:
        """
        print("V: " + str(value))

    def changeHorizontal(self, v):
        """
        垂直滑块的值发生改变时
        :param v:
        :return:
        """
        if v == 0:
            # 由于下列的操作会产生浮点型数据，所以在计算坐标的时候总有偏差
            # 所以在 v = 0， 也就是滑动条回到最开始的地方，
            # self.textContainer对应方向上的坐标也归0，那样就不会出问题了
            self.textContainer.move(0, self.textContainer.y())
            # 将当前值作为下一次的preValue
            self.hPreValue = v
            return

        # print("change H: {}".format(v))
        textW = self.textContainer.width()
        # 最大值（从0开始计算）
        maximum = self.hBar.maximum() + 1
        # 计算每次改变值对应 self.textContainer在水平方向上需要移动的尺寸
        stepW = textW / maximum
        # 当前值与上一次的值得差
        change = v - self.hPreValue

        move = int(self.textContainer.x() - stepW * change)

        self.textContainer.move(move, self.textContainer.y())
        # 将当前值作为下一次的preValue
        self.hPreValue = v

    def changeVertical(self, v):
        """
        水平滑块的值发生改变时
        :param v:
        :return:
        """
        if v == 0:
            # 由于下列的操作会产生浮点型数据，所以在计算坐标的时候总有偏差
            # 所以在 v = 0， 也就是滑动条回到最开始的地方，
            # self.textContainer对应方向上的坐标也归0，那样就不会出问题了
            self.textContainer.move(self.textContainer.x(), 0)
            self.vPreValue = v
            return

        # print("change V: {}".format(v))
        textH = self.textContainer.width()
        # 最大值（从0开始计算）
        maximum = self.vBar.maximum() - self.vBar.minimum() + self.vBar.pageStep()
        # 计算每次改变值对应 self.textContainer在水平方向上需要移动的尺寸
        stepH = float(textH) / maximum
        # 当前值与上一次的值得差
        change = v - self.vPreValue
        move = int(self.textContainer.y() - stepH * change)

        self.textContainer.move(self.textContainer.x(),  move)
        # 将当前值作为下一次的preValue
        self.vPreValue = v

    def sizeChange(self, obj: QLabel):
        print("size change:{},{} ".format(obj.width(), obj.height()))



