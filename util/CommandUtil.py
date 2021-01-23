
from util.FileUtil import FileUtil


class Process(object):

    def __init__(self, name=None, id=0, stageName=None, stage=0, ramUsage=0):
        super().__init__()
        self.__name = ""
        self.__id = 0
        self.__stageName = ""
        self.__stage = 0
        self.__ramUsage = 0

    def getName(self):
        return self.__name

    def getId(self):
        return self.__id

    def getStageName(self):
        return self.__stageName

    def getStage(self):
        return self.__stage

    def getRamUsage(self):
        return self.__ramUsage

    def toString(self):
        return "name={}, id={}, stageName={}, stage={}, ramUsage={}".format(
            self.__name, self.__id, self.__stageName, self.__stage, self.__ramUsage)


class CommandUtil(FileUtil):

    def __init__(self):
        super().__init__()
        self.__lines = []

    def setFilePath(self, path):
        super().setFilePath(path)
        self.__lines = super().read()

    def getProcessList(self):
        for line in self.__lines:
            _line = str(line)
            print(_line)
            if _line is None or len(_line) == 0 or _line == "":
                continue
            # split = _line.split(" ")
            inx = _line.index("exe")
            print("index: ", inx)


if __name__ == '__main__':
    util = CommandUtil()
    util.setFilePath(r"D:\WorkSpace\ProjectSpace\PycharmSpace\ProjectSpace\AI\resource\temp\ProcessList.txt")
    util.getProcessList()
