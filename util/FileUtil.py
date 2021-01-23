"""
@author kamiyong
@date 2021-1-20 10:14:46
@description 文件工具类
"""


class FileUtil(object):

    def __init__(self):
        super().__init__()
        self.filePath = None

    def setFilePath(self, path):
        self.filePath = path

    def read(self):
        with open(self.filePath) as f:
            return f.readlines()
