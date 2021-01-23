
import os
from util.ConfigUtil import ConfigUtil
from util.Logger import recordLog


if __name__ == '__main__':
    try:
        print("parent: ", os.getcwd())
        util = ConfigUtil()
        util.setFilePath(r"resource\config\AppConfig.txt")
        config = util.getConfig()
        recordLog("Config: " + config.toString())
        cmd = r"{}{}&python {}\main.py".format(config.getCondaPath(), config.getLibName(), os.getcwd())
        recordLog("cmd: " + cmd)
        cmd = cmd.replace("\n", " ")
        info = os.system(cmd)
        recordLog(info)
    except Exception as e:
        recordLog(e)
        raise e

