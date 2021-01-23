"""
@author kamiyong
@date 2021-1-20 10:19:16
@description Txt文件读取工具类
"""
from util.FileUtil import FileUtil
from bean.Text import Text


class TextFileUtil(FileUtil):

    def __init__(self):
        super().__init__()
        self.content = None

    def readTxt(self, filePath,  text: Text):
        super().setFilePath(filePath)
        self.content = super().read()




