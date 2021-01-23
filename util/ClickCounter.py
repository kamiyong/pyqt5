"""
    点击计数器
"""
import time


class Count(object):

    def __init__(self):
        # 精确到毫秒
        self.time = int(time.time() * 1000)

    def getTime(self):
        return self.time


class ClickCounter(object):

    def __init__(self):
        self.__container = []

    def record(self):
        self.__container.append(Count())

    def getCount(self, overTime):
        """
        查看在某个时间范围内的点击次数
        :param overTime: 超时时间（毫秒）
        :return: int
        """
        count = 0
        for counter in self.__container:
            # 精确到毫秒
            t = (time.time() * 1000)
            if t - counter.getTime() < overTime:
                count += 1
        return count

    def clear(self):
        """
        清空
        :return:
        """
        self.__container.clear()
