"""
@author kamiyong
@date 2021-1-16 14:07:30
@description 随机字符串
"""
from random import random
import time


class RandomString(object):
    """
    随机字符串
    """
    def __init__(self, length=0):
        self.__length = length
        self.__container = []
        self.__text = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def random(self, length=0):
        """
        随机字符串
        :param length: 字符串长度
        :return: 字符串
        """
        if length == 0:
            return ""

        seek = int(random() * length)

        for i in range(seek):
            index = int(random() * 52)
            # print("index: " + str(index))
            self.__container.append(self.__text[index])

        res = "".join(self.__container)
        self.__container.clear()
        return res

    def random_(self):
        """
        随机 随机长度的字符串
        :return: 字符串
        """
        t = str(time.time())
        # 获取当前时间戳，然后在其中随机找两个数字
        # 第一个作为个位， 第二个作为十位，第三个做百位，合成一个三位数字 A
        # 最后在随机随机这个三位数字
        f = int(random() * len(t) + 1)
        s = int(random() * len(t) + 1)
        t = int(random() * len(t) + 1)
        A = (t * 100 + s * 10 + f)
        length = int(random() * A + 1)

        for i in range(length):
            index = int(random() * 52)
            self.__container.append(self.__text[index])

        res = "".join(self.__container)
        self.__container.clear()
        return res
