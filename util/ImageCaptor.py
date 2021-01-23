"""
@author kamiyong
@date 2021-1-19 10:08:18
@description 图片获取工具类
"""
import os
import time
from PIL import ImageGrab


class ImageCaptor(object):
    """
    图片获取
    """
    def __init__(self):
        self.__parent = "/resource/temp/TempImage/"
        self.__path = os.getcwd()
        self.__timeFormat = "%y-%m-%d_%H-%M-%S"
        self.__image = None
        self.__imageTempPath = None

    def capture(self):
        """
        截取屏幕作为图片
        :return: 返回存儲路徑， 原始截屏圖片
        """
        self.__image = ImageGrab.grab()
        self.__imageTempPath = self.__path + self.__parent + time.strftime(self.__timeFormat) + ".jpg"
        # 暂存图片
        # 真正的永久存储图片请参照{@link self.save()}
        self.__image.save(self.__imageTempPath)

        return [self.__imageTempPath, self.__image]

    def save(self, barcode, result):
        """
        永久存储图片
        :param barcode: 条码
        :param result: 结果
        :return: no return
        """
        # 存储照片的文件格式为：当前项目执行路径/image/当前日期
        # 如：D:/Test/image/2020-11-10/OK
        if self.__image is None:
            return
        image_path = self.__path + "/resource/img/" + time.strftime("%y-%m-%d") + "/" + result
        # 文件夹不存在则新建一个
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        # 图片名称以条码加当前日期（精确到秒）的格式
        save_path = image_path + "/" + barcode + "_" + time.strftime("%y-%m-%d_%H-%M-%S") + ".jpg"
        # 此处才是真正存储图片到本地
        self.__image.save(save_path)

    def delete(self):
        """
        删除暂存的图片
        :return: no return
        """
        if self.__imageTempPath is not None:
            if os.path.exists(self.__imageTempPath):
                os.remove(self.__imageTempPath)
