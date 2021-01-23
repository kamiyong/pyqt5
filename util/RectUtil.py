
"""
@author kamiyong
@date 2021-1-20 10:19:16
@description Txt文件读取工具类
"""
from util.FileUtil import FileUtil
from bean.Rect import Rect


class RectUtil(FileUtil):

    def __init__(self):
        super().__init__()
        self.__content = None
        self.__rectList = []

    def setFilePath(self, filePath):
        super().setFilePath(filePath)
        self.__content = super().read()

    def __split(self):
        """
        拆分内容
        :return:
        """
        for line in self.__content:
            if line is None or line == "":
                continue
            rect = Rect()
            item = line.split(",")
            for it in item:
                element = it.split("=")
                rect.setValue(element[0], element[1])
            self.__rectList.append(rect)

    def getRectList(self):
        self.__split()
        return self.__rectList
