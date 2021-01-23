"""
@author kamiyong
@date 2021-1-20 10:31:56
@description 矩形框实体类
"""
from bean.Text import Text


class Rect(Text):

    __ID = "id"
    __NAME = "name"
    __LEFT = "left"
    __TOP = "top"
    __RIGHT = "right"
    __BOTTOM = "bottom"
    __CREATE_TIME = "create_time"

    def __init__(self, id=None, name=None, left=None, top=None, right=None, bottom=None, createTime=None):
        super(Rect, self).__init__()
        self.__id = id
        self.__name = name
        self.__left = left
        self.__top = top
        self.__right = right
        self.__bottom = bottom
        self.__createTime = createTime

    def toString(self):
        return "id={}, name={}, left={}, top={}, right={}, bottom={}, createTime={}".format(
            self.__id, self.__name, self.__left, self.__top, self.__right, self.__bottom, self.__createTime)

    def center(self):
        """
        获取矩形中心点
        :return:
        """
        return [(self.__right - self.__left) / 2, (self.__bottom - self.__top) / 2]

    def position(self):
        """
        返回四个坐标点
        :return:
        """
        return [self.__left, self.__top, self.__right, self.__bottom]

    def name(self):
        return self.__name

    def width(self):
        return self.__right - self.__left

    def height(self):
        return self.__bottom - self.__top

    def setValue(self, key, value):
        if key == Rect.__ID:
            self.__id = int(value)
        elif key == Rect.__NAME:
            self.__name = value
        elif key == Rect.__LEFT:
            self.__left = int(value)
        elif key == Rect.__TOP:
            self.__top = int(value)
        elif key == Rect.__RIGHT:
            self.__right = int(value)
        elif key == Rect.__BOTTOM:
            self.__bottom = int(value)
        elif key == Rect.__CREATE_TIME:
            self.__createTime = value
