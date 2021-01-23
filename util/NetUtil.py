"""
    @author kamiyong 2020-12-23 15:14:39
    iAUTO 工具类

"""
import requests
from requests.adapters import HTTPAdapter

from constant.Global import ip, recorder_info_to_iauto

from util.CommonUtil import getMacAddress, getTimeNow
from util.Logger import printInfo
from util.ExcelUtil import read_station

"""
FA AI工站測試接口:
http://172.18.241.58/autoaw5?c=QUERY_RECORD&sn=DVPZM005MLTK
&sfc_station=AI&mac_address=00:00:00:00&line=M01-4FT-01&p=ROUTE_CHECK


PASS測試記錄上傳：
http://172.18.241.58/autoaw5?sn=JFLKASJDFLKJAS&c=ADD_RECORD&test_station_name=AI-AI&station_id=FXCD_M01-4FT-01_01_AI-AI
&mac_address=00:00:00:00:00&stop_time=2011-04-19%2008:00:00
&result=PASS&start_time=2011-04-19%2007:59:28&line=M01-4FT-01&sfc_station=AI


FAIL測試記錄上傳：
http://172.18.241.38/autoaw5?sn=JFLKASJDFLKJAS&c=ADD_RECORD&test_station_name=AI-AI&station_id=FXCD_M01-4FT-01_01_AI-AI
&mac_address=00:00:00:00:00&stop_time=2011-04-19%2008:00:00&result=FAIL
&start_time=2011-04-19%2007:59:28&list_of_failing_tests=Wrong&failure_message=Wrong
"""

ip_iAUTO_query_head = 'http://' + ip + '/autoaw5?c=QUERY_RECORD&sn='
ip_iAUTO_query_body_1 = '&sfc_station=AI&mac_address='
ip_iAUTO_query_body_2 = '&line=M01-3FA-01&p=ROUTE_CHECK'

ip_iAUTO_add_head = 'http://' + ip + '/autoaw5?c=ADD_RECORD&sn='
ip_iAUTO_add_body = '&test_station_name=AI-AI&station_id=FXLH_G15-6FT-06A_06_AI-AI&mac_address='
start_time = '&start_time='
stop_time = '&stop_time='
line = '&line=M01-3FA-01&sfc_station=AI'
result_pass = '&result=PASS'
result_fail = '&result=FAIL&list_of_failing_tests='
message = '&failure_message='

s = requests.Session()
s.mount('http://', HTTPAdapter(max_retries=2))
s.mount('https://', HTTPAdapter(max_retries=2))


# 查询机种信息
def get_res_message(barcode):
    url = ip_iAUTO_query_head + barcode + ip_iAUTO_query_body_1 \
          + getMacAddress() + ip_iAUTO_query_body_2
    printInfo("請求機台信息 url: " + url)

    try:
        with s.get(url, timeout=5, stream=True) as r:
            req = r.text
            return req
    except Exception as e:
        printInfo("請求機台信息異常： " + str(e))
        return str(e)


# 判断正确调用url
def get_res_ok(barcode, time_start):
    if not recorder_info_to_iauto:
        return "上傳已關閉"
    url = ip_iAUTO_add_head + barcode + ip_iAUTO_add_body \
          + getMacAddress() + start_time + time_start + stop_time \
          + getTimeNow() + result_pass + line
    printInfo("上傳OK結果url： " + url)
    try:
        with s.get(url, timeout=5, stream=True) as r:
            req = r.text
            printInfo("上傳OK結果到iAUTO返回信息：" + req)
            return req
    except Exception as e:
        printInfo("上傳OK結果異常：" + str(e))
        return str(e)


# 判断错误调用URL
def get_res_fall(barcode, begin_time, num, fall_msg):
    if not recorder_info_to_iauto:
        return "上傳已關閉"
    url = ip_iAUTO_add_head + barcode + ip_iAUTO_add_body + getMacAddress() + start_time + \
          begin_time + stop_time + getTimeNow() + result_fail + num + \
          message + fall_msg
    printInfo("上傳NG結果url： " + url)
    try:
        with s.get(url, timeout=5, stream=True) as r:
            req = r.text
            printInfo("上傳NG結果到iAUTO返回信息：" + req)
            return req
    except Exception as e:
        printInfo("上傳NG結果異常：" + str(e))


def get_route_station(iauto_result):
    """
    路由错误的时候, 根据路由信息查询对应的工站名称
    :param iauto_result
    """
    # 参数格式： 0 SFC_OK ROUTE_CHECK=Route Error Go to:(AK-Plug Alert Flex B2B to SIP and press),CATEGORY_KEY=N140S

    # 其他路由错误的格式：
    # (1) -- 0 SFC_OK ROUTE_CHECK=Invalid Wip no
    # (2) -- 0 SFC_OK ROUTE_CHECK=Work order has been closed
    # 这种格式的信息就不必去查表了
    if iauto_result.find("Go to") == -1:
        sp = iauto_result.split("=")[1]
        if sp is None:
            return "路由錯誤"
        return "路由錯誤: " + str(sp)

    substring = iauto_result[iauto_result.index("(") + 1: iauto_result.index(")")]

    split = substring.split("-")
    print("split: ", split)
    if split[0] is None or split[1] is None:
        return substring
    read = read_station(split[0], split[1])
    # 未找到
    if read is None:
        return "未知工站名"
    return "Go to: " + str(read)
