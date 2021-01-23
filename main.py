"""
Created on 2021-1-12 10:44:16
@author: kamiyong
@file: main
@description: 启动类
"""

import sys
import os
import time
from PyQt5.QtWidgets import QApplication
from activity.MainMiniActivity import MainWindow
from constant import Context
from util.Resource import Resource
from helper.MainHelper import MainHelper
from threading import Thread

sys.path.append(os.getcwd())
res = Resource()
helper = MainHelper()


def activityCloseCallback():
    helper.destroy()
    print("destroy thread!")


def runTask():
    while 1:
        time.sleep(5)
        helper.barcodeCallback("DJKDJKLDJLKDL")


def test():
    Thread(target=runTask).start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    mainQss = res.readCssAsStream("mini.qss")
    window.setStyleSheet(mainQss)
    window.setCloseWindowCallback(activityCloseCallback)

    helper.setBarcodeET(window.getBarcodeEditText())
    helper.setDisplayBox(window.getDisplayBox())
    helper.setResultBox(window.getResultBox())
    helper.start()
    window.display()

    print(Context.ctx.getCtx())
    # test()

    sys.exit(app.exec_())
