from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from MODEL.luo.transforms import transforms
from MODEL.luo.model import MainModel
import cv2
import numpy as np
import math
import time

device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = MainModel()
model_dict = model.state_dict()
pretrained_dict = torch.load('MODEL/luo/net_model/weights_20_139_1.0000_1.0000.pth')
pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.train(False)


def model_predict(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    input_img = transform(img).unsqueeze(0)
    inputs = input_img.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        outputs_pred = outputs[0] + outputs[1][:, 0:3] + outputs[1][:, 3:2 * 3]
        _, predicted = torch.max(outputs_pred, 1)
        classes = predicted.cpu().numpy()
        return classes[0]


def get_white(img):
    # # �զ�
    lower_white = np.array([0, 0, 170])
    upper_white = np.array([180, 45, 255])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask


def get_bound(cnts):
    allarea = np.concatenate(cnts)
    x, y, w, h = cv2.boundingRect(allarea)
    return x, y, w, h


def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def compute(img, min_percentile, max_percentile):
    """?�����?�A�ت��O�h��?1������???���ݱ`��?"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel


def get_lightness(src):
    # ?��G��
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness


def aug(src):
    """?���G�׼W?"""
    if get_lightness(src) > 130:
        print("?���G�ר�?�A�����W?")
    # ��?�����?�A�h�������Ȥ���?�ݱ`�ȡA??����?�i�H�ۤv�t�m�C
    # ��p1������?��?��b0��255�W�����ȡA���O??�W�����ȥD�n�b0��20?�C
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # �h�������??���~����
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # ?�����??�Ԧ���0��255�A?�����F255*0.1�O255*0.9�O�]?�i��?�X?�����ȷ��X����?�A�ҥH�̦n���n?�m?0��255�C
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out


def new_image_processing(image_path, ww, hh):
    '''
    :param image_path: ?������?
    :param ww: 0
    :param hh: 260  # big, 195  # small
    :return: ���ŦZ��?���Asize = 52 * 25
    '''
    img = cv2.imread(image_path)
    # cv2.imshow("img", img)
    # cv2.waitKey()
    img = img[:, :img.shape[1] // 2]
    #img = aug(img)
    # cv2.imshow('aug', img)
    guass = cv2.GaussianBlur(img, (5, 5), 0)
    wmask = get_white(guass)
    #cv2.imshow('wmask', wmask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ero = cv2.erode(wmask, kernel)
    cnts, _ = cv2.findContours(ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_cnts = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if cnt_area(cnt) > 2500 and cnt_area(cnt) < 5500 and w < 150 and h < 120 and h > w:
            new_cnts.append(cnt)
    if len(new_cnts) > 1:
        new_cnts.sort(key=cnt_area, reverse=True)
        y0 = 0
        cnt0 = new_cnts[0]
        for ncnt in new_cnts:
            x, y, w, h = cv2.boundingRect(ncnt)
            if y0 > y and x > 240 and x < 300:
                cnt0 = ncnt
                y0 = y
        new_cnts = cnt0

    cut_img = img
    if len(new_cnts) > 0:
        x1, y1, w1, h1 = get_bound(new_cnts)
        cut_img = img[y1 + h1 - 25 - hh:y1 + h1 - hh, x1 + w1 - 52 - ww:x1 + w1 - ww]  # new
        #cv2.imshow('cut', cut_img)
        #cv2.waitKey()
    return cut_img


# ?��?������t
def img_bright_mean(img):
    img_mean = np.mean(img)
    return img_mean


# �G��??
def balance_light(img, img_bright_mean, range, alpha=1.5):
    '''
    @param img hsv?������v�q�D
    @param img_std:img �� ?���G�ק���t
    @std ?�㧡��t
    @alpha ��?????�Aalpha�V�j�A�ĪG�V��?
    '''
    if img_bright_mean > range[1]:
        inc = int(alpha * math.pow(int(math.fabs(range[1] - img_bright_mean)), 1.2))
        img[img < inc] = inc
        img = img - inc
    elif img_bright_mean < range[0]:
        inc = int(alpha * math.pow(int(math.fabs(range[0] - img_bright_mean)), 1.2))

        img[img > 255 - inc] = 255 - inc
        img = img + inc
    return img


def adjust_img_brightness(img):
    # cv2.imshow('src', img)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # ?��?���G�ק���
    img_mean = img_bright_mean(hsv_img[:, :, 2])
    # print('?������t ', img_mean)
    # �G�ק��ȭS?
    # VUE_MEAN_RANGE = (125, 140)
    VUE_MEAN_RANGE = (82, 90)

    v_channel = balance_light(hsv_img[:, :, 2], img_mean, VUE_MEAN_RANGE, alpha=0.5)
    hsv_img[:, :, 2] = v_channel
    bgr_brighter = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    #cv2.imshow('cut', bgr_brighter)
    #cv2.waitKey()
    return bgr_brighter


def cut_data_test(watch_size, img_path):
    if watch_size == 'B':
        ww = 0
        hh = 260
    else:
        ww = -2
        hh = 195
    start = time.time()
    img = new_image_processing(img_path, ww, hh)
    cost = (time.time() - start) * 1000
    print("new_image_processing:", cost, "ms")

    img = cv2.resize(img, (52, 25))

    start = time.time()
    img = adjust_img_brightness(img)
    cost = (time.time() - start) * 1000
    print("adjust_img_brightness:", cost, "ms")

    start = time.time()

    reslut = model_predict(img)
    #cv2.imshow('img', img)
    #cv2.waitKey()
    cost = (time.time() - start) * 1000
    print("model_predict:", cost, "ms")

    return reslut


if __name__ == '__main__':
    import os, time

    img_path = r'D:\PY_scipty\DCL-master_1\DCL-master\img\t\1'
    for file in os.listdir(img_path):
        file_path = os.path.join(img_path, file)
        start_time = time.time()
        result = model_predict(file_path)
        print(result)
        if result != 2:
            print(file_path)
        end_time = time.time()
        print('use_tim: ', end_time - start_time)
