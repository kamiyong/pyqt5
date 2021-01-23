"""
@author kamiyong
@date 2021-1-19 10:46:38
@description 日志记录类
"""
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='resource/log/AW_log.log', level=logging.INFO, format=LOG_FORMAT)


def printInfo(text):
    print(text)


class Logger:

    def __init__(self, name, level):
        super()

    def record(self, text):
        # self.info(text)
        logging.info(text)


logger = Logger("", "DEBUG")


def recordLog(text):
    logger.record(text)
