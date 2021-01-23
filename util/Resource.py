
"""
Created on 2021-1-12 11:43:55
@author: kamiyong
@file: Resource
@description: 资源管理类
"""
import os


class Resource(object):

    def __init__(self):
        # 获取当前项目执行路径
        self.__parent = os.getcwd()

    def readImage(self, imageName):
        return r"{}\{}\{}".format(self.__parent, r"resource\image", imageName)

    def readConfig(self):
        return r"{}\{}\{}".format(self.__parent, r"resource\config", "AppConfig.txt")

    def readCss(self, cssName):
        return r"{}\{}\{}".format(self.__parent, r"resource\css", cssName)
    
    def readCssAsStream(self, cssName):
        css = self.readCss(cssName)
        if not os.path.exists(css):
            print("File '" + cssName + "' is not exists.")
            return None
        with open(css) as f:
            return f.read()


res = Resource()

