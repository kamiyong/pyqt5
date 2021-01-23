"""
@author
"""
from util.FileUtil import FileUtil
from bean.Config import Config


def emptyString(string):
    return string is None or len(string) == 0 or string == ""


class ConfigUtil(FileUtil):

    def __init__(self):
        super().__init__()
        self.__container = None

    def setFilePath(self, path):
        super().setFilePath(path)
        self.__container = super().read()

    def getConfig(self):
        app = Config()
        for line in self.__container:
            print("line: ", line)
            if emptyString(line) or line[0] == "#":
                continue
            item = line.split("=")
            if len(item) < 2:
                continue
            app.setValue(item[0], item[1])
        return app
