"""
@author kamiyong
@date 2021-1-19 10:43:06
@description 公共工具类
"""
import uuid
import datetime
import time


# mac地址
def getMacAddress():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e + 2] for e in range(0, 11, 2)])


# 系统时间
def getTimeNow():
    time_before = datetime.datetime.now()
    time_str = datetime.datetime.strftime(
        time_before, '%Y-%m-%d %H:%M:%S').replace(' ', '%20')
    return time_str


def getCurrentTime():
    return time.time()


class TimeD:
    # 时间范围的开始时间(24小时制的早上)
    start_time_am_24 = "07:40:00"

    # 时间范围的结束时间(24小时制的早上)
    end_time_am_24 = "07:50:00"

    # 时间范围的开始时间(24小时制的下午)
    start_time_pm_24 = "19:40:00"
    # start_time_pm_24 = "15:30:00"

    # 时间范围的结束时间(24小时制的下午)
    end_time_pm_24 = "19:40:00"
    # end_time_pm_24 = "15:50:00"

    # 时间范围的开始时间(12小时制)
    start_time_12 = "07:40:00"
    # start_time_12 = "03:22:00"

    # 时间范围的结束时间(12小时制)
    end_time_12 = "07:50:00"
    # end_time_12 = "03:50:00"
