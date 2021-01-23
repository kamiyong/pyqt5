"""
@author kamiyong
@date 2021-1-21 09:19:30
@description 应用程序配置类
"""
from bean.Text import Text


class Config(Text):

    __CONDA_PATH = "conda_path"
    __LIB_NAME = "lib_name"

    def __init__(self):
        super().__init__()
        self.__condaPath = None
        self.__libName = None

    def toString(self):
        return "condaPath={}, libName={}".format(self.__condaPath, self.__libName)

    def setValue(self, key, value):
        if key == Config.__CONDA_PATH:
            self.__condaPath = value
        elif key == Config.__LIB_NAME:
            self.__libName = value

    def getCondaPath(self):
        return self.__condaPath

    def getLibName(self):
        return self.__libName
