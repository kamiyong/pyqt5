
"""
Created on 2021-1-12 11:04:17
@author: kamiyong
@file: Context
@description: 上下文类，用于存储全局参数，在应用程序的任意地方都能使用
"""


class Context(object):
    """
        上下文类
    """
    __BOX = {}

    def put(self, key, value):
        Context.__BOX.update({key: value})

    def get(self, key):
        return Context.__BOX.get(key)

    def clear(self):
        Context.__BOX.clear()

    def remove(self, key):
        Context.__BOX.pop(key)

    def getCtx(self):
        return Context.__BOX.copy()


class Key(object):
    """
        键值类
    """
    def __init__(self):
        self.key = "None"

    default = "key"
    # 屏幕可用尺寸
    SCREEN_AVAIL_SIZE = "screen_avail_size"
    # 屏幕真实尺寸
    SCREEN_REAL_SIZE = "screen_real_size"
    # 当前尺寸
    WINDOW_INIT_SIZE = "window_init_size"
    # 窗口头部高度
    WINDOW_HEAD_HEIGHT = "window_head_height"


# 此处的实例时全局通用的
ctx = Context()
