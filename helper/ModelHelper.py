# update time: 2020-12-02 09:20
# update author: YongYe

import cv2
import sys
import os
import time

from MODEL.luo.new_AW_test import cut_data_test
from MODEL.zhang.crew_check import val
from MODEL.wang.predict_blade import get_roi_data_balde, model_predict_blade


# Feed bar用料异常
# 140 WIFI
# 142 3G
def check_luo(size, image_path):
    res = cut_data_test(size, image_path)
    return "140" if res == 0 or (res == 1 and size == "S") else "142"


# Blade偏位异常
def check_wang(size, image_path):
    model_path = r'MODEL/wang/blade_rgb.pth'
    template_img_path = r"MODEL/wang/template_blade.jpg"
    roi = get_roi_data_balde(image_path, watch_size=size)
    res = model_predict_blade(roi, model_path, template_img_path)
    if res == 1:
        return 0
    else:
        return 1


# 螺丝检测
def check_zhang(size, image_path):
    img = cv2.imread(image_path)
    res1 = val(img, r'MODEL/zhang/p1.model', 1, size)
    res2 = val(img, r'MODEL/zhang/p2.model', 2, size)
    if res1 == 1 and res2 == 1:
        return 0
    else:
        return 1


def result_all(size, image_path):
    result_list = []

    start = time.time()
    result_luo = check_luo(size, image_path)
    cost = int((time.time() - start) * 1000)
    print("Time cost to check luo: ", cost, "ms")

    start = time.time()
    result_wang = check_wang(size, image_path)
    cost = int((time.time() - start) * 1000)
    print("Time cost to check wang: ", cost, "ms")

    start = time.time()
    result_zhang = check_zhang(size, image_path)
    cost = int((time.time() - start) * 1000)
    print("Time cost to check zhang: " + str(cost) + "ms")

    print("wang:", result_wang)
    print("luo:", result_luo)
    print("zhang:", result_zhang)

    result_list.append(result_luo)
    result_list.append(result_wang)
    result_list.append(result_zhang)

    return result_list


"""
0 OK
1 NG
"""

if __name__ == '__main__':
    print("image-path", os.path.exists(r"E:\TestSpace\Python\Web\Test\MODEL\wang\140b.jpg"))
    print(result_all('B', r"E:\TestSpace\Python\Web\Test\MODEL\wang\140b.jpg"))
