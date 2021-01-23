"""
@author kamiyong
@date 2021-1-19 09:59:05
@description 业务处理类
"""
import re
import time
from threading import Thread

from view.TextView import TextView

from helper.ModelHelper import result_all

from util.ImageCaptor import ImageCaptor
from util.CommonUtil import getTimeNow, getCurrentTime
from util.Logger import printInfo, recordLog
from util.NetUtil import get_res_ok, get_res_fall, get_res_message, get_route_station


class MainHelper(Thread):
    """
    业务处理类
    """

    def __init__(self):
        super().__init__()
        # 条码输入框
        self.barcodeET = None
        # 图片获取工具类
        self.imageCaptor = ImageCaptor()
        # 展示流程的組件
        self.displayBox = None
        # 展示最终结果的组件
        self.resultBox = None
        # 任务是否完成的标志位
        self.finish = True
        # 条码
        self.barcode = None
        # 是否正在识别
        self.action = True
        self.running = True
        # 流程编号
        self.number = 0
        #
        self.finalResult = ""

    def setDisplayBox(self, box):
        self.displayBox = box

    def setResultBox(self, box):
        self.resultBox = box

    def setBarcodeET(self, barcodeET: TextView):
        """
        设置条码输入框的实例
        :param barcodeET {@link view.TextView}
        :return: no return
        """
        self.barcodeET = barcodeET
        self.barcodeET.setCallback(self.barcodeCallback)

    def show(self, content):
        """
        展示 流程
        :param content: 展示的具体信息
        :return:no return
        """
        self.number += 1
        self.displayBox.append("{}.{}".format(self.number, content))

    def showResult(self, result, Pass=True):
        """
        展示最终结果，如： NG OK or ...
        :param Pass: 是否Pass
        :param result: 最终的结果
        :return:  no return
        """
        fontSize = "70px"
        length = len(result)
        if length < 5:
            fontSize = "70px"
        elif 5 <= length < 8:
            fontSize = "40px"
        elif 8 <= length < 12:
            fontSize = "30px"
        elif 12 <= length:
            fontSize = "20px"

        color = "#00FF00" if Pass else "#FF0000"
        style = "#resultBox {border: 2px solid rgba(81, 190, 247, 0.5);font-size: " \
                + fontSize + ";color: " + color + ";font-family: 'Microsoft YaHei';}"

        self.resultBox.setStyleSheet(style)
        self.resultBox.setText(result)

    def isFinish(self):
        return self.finish

    def endTask(self):
        """
        结束任务并且重置标志位
        :return:
        """
        self.number = 0
        self.finish = True

    def clear(self):
        self.resultBox.setText("")
        self.displayBox.setText("")

    def barcodeCallback(self, barcode: str):
        """
        条码输入框回调函数， 以该函数触发AI动作
        :param barcode: 条码
        :return: no return
        """

        # 当条码不为空并且上一次流程已经结束，才能进行这一次的流程
        if barcode is None:
            self.show("获取条码失败！")
            return
        length = len(barcode)
        if barcode[length - 1] == "\n":
            barcode = barcode[0: length - 1]
        printInfo("barcode:" + barcode)
        if self.finish:
            self.clear()
            self.barcodeET.setText("")
            self.finish = False
            self.barcode = barcode
            self.action = False
            recordLog("")
            recordLog("************************")
            self.show("開始...")
            recordLog("開始...")
            self.show("獲取到條碼!")
            self.show("條碼：" + str(barcode))
            recordLog("獲取到條碼：" + str(barcode))
        else:
            self.barcodeET.setText("")

    def startTask(self):
        """
        开始任务
        :return: no return
        """
        try:
            startTime = getTimeNow()
            self.show("開始獲取圖片！")
            recordLog("開始獲取圖片！")
            # 获取图片
            imagePath = self.imageCaptor.capture()[0]
            if imagePath is None:
                self.show("獲取圖片失敗！")
                recordLog("獲取圖片失敗！")
                return

            self.show("獲取圖片成功！")
            recordLog("獲取圖片成功！")
            self.show("開始查詢機台信息！")
            recordLog("開始查詢機台信息！")
            # resultIAuto = get_res_message(self.barcode)
            # iAUTO正常请求返回格式
            resultIAuto = '0 SFC_OK ROUTE_CHECK=OK,CATEGORY_KEY=N140S'
            # 错误格式
            # resultIAuto = '0 SFC_OK ROUTE_CHECK=Wip has been produced, not in work'
            # resultIAuto = '0 SFC_OK ROUTE_CHECK=Route Error Go to:(BA-Check comp3),CATEGORY_KEY=N140S'
            # resultIAuto = '0 SFC_OK ROUTE_CHECK=Route Error Go to:
            # (CY-Check battery capacity/confirm SN/shut down),CATEGORY_KEY=N140S'
            recordLog("查詢機台返回信息：" + str(resultIAuto))
            if resultIAuto is None:
                self.show("iAuto 請求失敗！")
                recordLog("iAuto 请請求失敗！")
                return

            if resultIAuto[0] == '0':
                ROUTE_CHECK = re.findall(r'ROUTE_CHECK=(.*)', resultIAuto)[0]
                if 'OK' in ROUTE_CHECK:
                    self.show("查詢機台信息成功！")
                    recordLog("查詢機台信息成功！")
                    CATEGORY_KEY = re.findall(r'CATEGORY_KEY=(.*)', resultIAuto)[0]
                    product = CATEGORY_KEY[-2]
                    size = CATEGORY_KEY[-1]
                    if product == '0':
                        product = '140'
                    else:
                        product = '142'

                    # 识别结果
                    # luo: Feed bar用料异常
                    # wang: Blade偏位异常
                    # zhang: 螺丝检测
                    # 0: OK, 1: NG
                    #                  luo,  wang, zhang
                    # result_model = ['140',    0,     1]
                    self.show("開始識別！")
                    recordLog("開始識別！")
                    startCheckTime = getCurrentTime()
                    result = result_all(size, imagePath)
                    cost = int((getCurrentTime() - startCheckTime) * 1000)
                    self.show("識別完成！ 耗时：" + str(cost) + "ms")
                    recordLog("識別完成！ 耗時：" + str(cost) + "ms")
                    result = ['142', 1, 1]
                    recordLog(result)
                    judge_result, show_msg, upload_msg = judgement(product, result)
                    self.finalResult = judge_result
                    self.show("判定結果：" + judge_result)
                    recordLog("判定結果：" + judge_result)
                    if judge_result == "OK":
                        self.showResult(judge_result)
                        upload_res = get_res_ok(self.barcode, startTime)
                        self.formatUploadInfo(upload_res)
                    else:
                        self.showResult(judge_result, False)
                        self.show("詳細：" + show_msg)
                        upload_res = get_res_fall(self.barcode, startTime, "1", upload_msg)
                        self.formatUploadInfo(upload_res)
                else:
                    self.show("請求iAUTO，路由錯誤！")
                    recordLog("請求iAUTO，路由錯誤！")
                    self.show(resultIAuto)
                    show = get_route_station(resultIAuto)
                    self.showResult(show, False)
                    upload_res = get_res_fall(self.barcode, startTime, '2', 'ROUTE_CHECK-fail')
                    self.formatUploadInfo(upload_res)
                    self.finalResult = "ROUTE_CHECK-fail"
            else:
                self.show("請求iAUTO失敗: " + str(resultIAuto))
                recordLog("請求iAUTO失敗: " + str(resultIAuto))
        except Exception as e:
            self.show("程序異常：" + str(e))
            recordLog("程序異常：" + str(e))
            raise e
        finally:
            self.imageCaptor.delete()
            self.imageCaptor.save(self.barcode, self.finalResult)
            self.show("結束！")
            recordLog("結束！")
            self.endTask()

    def formatUploadInfo(self, upload_res):
        if upload_res == "0 SFC_OK":
            self.show("上傳成功！")
            recordLog("上傳成功！")
        else:
            self.show("上傳異常：" + upload_res)
            recordLog("上傳異常：" + upload_res)

    def run(self) -> None:
        while self.running:
            # 不在识别状态的情况下才进入识别
            # 然后重置标志位，等待本次识别完成之后才能进入下一次的识别动作
            if not self.action:
                self.action = True
                self.startTask()
            # self.barcodeCallback("DJDKLJDKLDJLK")
            # self.startTask()
            time.sleep(1)

    def destroy(self):
        """
        销毁该线程
        :return: no return
        """
        self.running = False


show_message = ["Feed bar用料異常", "Blade偏位異常", "螺絲檢測異常", "Coco Cowling偏位異常"]
upload_message = ["Product-fail", "Goldfinger-fail", "Screw-fail", "CocoCowling-fail"]


def judgement(product, result):
    """
    根据识别结果综合判断最终的结果
    :param product 产品类型
    :param result 识别结果
    :return
        1. "OK" or "NG"
        2. 展示信息
        3. 上传信息
    """
    # 是否通过标志位
    Pass = True
    # 展示信息
    s_m = ""
    # 上传信息
    u_m = ""
    # result: ['140', 0, 0, 0]
    res_len = len(result)
    for i in range(res_len):
        item = result[i]
        # i 是 0 的时候， result 不是为数字，需要进行特殊判断
        if i == 0:
            # 判断和产品类型是否一样
            if item == product:
                s_m += ""
                u_m += ""

            else:
                Pass = False
                s_m += show_message[i]
                u_m += upload_message[i]
        # i 不为 0, 则判断是否 等于0， 0 是ok， 1 是ng
        else:
            if item == 0:
                s_m += ""
                u_m += ""
            else:
                Pass = False
                s_m += ("," + show_message[i])
                u_m += ("," + upload_message[i])

    # 去掉首位的逗号
    if len(s_m) > 0 and s_m[0] == ",":
        s_m = s_m[1: len(s_m)]
        u_m = u_m[1: len(u_m)]

    if Pass:
        return "OK", "", ""
    else:
        return "NG", s_m, u_m
