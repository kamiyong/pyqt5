"""
Created on 2021-1-12 15:11:45
@author: kamiyong
@file: Theme
@description: 主题类，对应用程序的颜色背景，字体等进行管控
"""
SYSTEM_DEFAULT_BG_COLOR = "#0C172D"


def getBgColor(color):
    return "background-color: {};".format(color)
