# -*- coding:utf-8 -*-
import cv2.cv2 as cv2
import numpy as np
import random
import math


# 计算图像均方差
def img_bright_mean(img):
    '''
     @param img hsv图像中的v通道
    '''
    img_mean = np.mean(img)
    # print(img_std)
    return img_mean


# 亮度调节
def balance_light(img, img_bright_mean, range, alpha=1.5):
    '''
    @param img hsv图像中的v通道
    @param img_std:img 的 图像亮度均方差
    @std 标准均方差
    @alpha 倍数调节参数，alpha越大，效果越明显
    '''
    if img_bright_mean > range[1]:
        inc = int(alpha * math.pow(math.fabs(range[1] - img_bright_mean), 1.2))
        img[img < inc] = inc
        img = img - inc
    elif img_bright_mean < range[0]:
        inc = int(alpha * math.pow(math.fabs(range[0] - img_bright_mean), 1.2))

        img[img > 255 - inc] = 255 - inc
        img = img + inc
    return img


# imgpath= r'E:\C3407841\task\AWFeedbar\test\1203_testing_result\OK\H4HDTB7RQ07V_20-12-03_17-29-28.jpg'
# img = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
# img_std(img)

# def balance_light(img,std):

# 腐蚀
def eroded(src, size=(3, 3)):
    # image = cv2.resize(image_src,(600,400))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=size)
    # dilated = cv2.dilate(image,kernel)
    # cv2.imshow('dilated',dilated)
    erode = cv2.erode(src, kernel)
    # thresholded = cv2.adaptiveThreshold(erode,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
    # ret,thresholded = cv2.threshold(erode,150,255,cv2.THRESH_BINARY)
    # cv2.imshow('erode',erode)
    # cv2.imshow('src',src)
    # cv2.waitKey(0)
    return erode


# 膨胀
def dilated(src, size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=size)
    dilate = cv2.dilate(src, kernel)
    return dilate


def sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def normalize_image(image):
    # normalization
    img = image.astype(np.float32) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img


def normalize_gray_image(image):
    # normalization
    img = image.astype(np.float32) / 255
    img = (img - 0.5) / 0.2
    return img


def brightness_varify(image, rg=0.2):
    '''
    调节图像亮度,
    :param image: 
    :param range:range of brightness varifaction,between 0 and 1,image brightness will varify from -range to range.
    :return: 
    '''
    h, w, _ = image.shape
    ret = image.copy()
    bright_inc = (-1 + 2 * random.random()) * rg
    for i in range(h):
        for j in range(w):
            r = (int)(image[i, j, 0] * (1 + bright_inc))
            g = (int)(image[i, j, 1] * (1 + bright_inc))
            b = (int)(image[i, j, 2] * (1 + bright_inc))
            if r >= 255:
                ret[i, j, 0] = 255
            else:
                ret[i, j, 0] = r
            if g >= 255:
                ret[i, j, 1] = 255
            else:
                ret[i, j, 1] = g
            if b >= 255:
                ret[i, j, 2] = 255
            else:
                ret[i, j, 2] = b
    return ret


def denormalize_image(image):
    '''
    去归一化
    :param image:
    :return:
    '''
    img = image.astype(np.float32)
    img = ((img * std) + mean) * 255
    img = img.astype(np.uint8)
    return img


def sobel_extract(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)  # 1,0代表只计算x方向计算边缘
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)  # 0,1代表只在y方向计算边缘
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


# 倾斜矫正
def img_rotate(img, angle):
    '''
    @param:原始图像
    @param:旋转角度
    return dstimg,M
    '''
    h, w, = img.shape[0], img.shape[1]
    cx, cy = int(w / 2), int(h / 2)
    # 仿射变换矩阵
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    # cv2.imshow('rotate',rotated)
    # cv2.waitKey()
    return rotated, M

# imgpath =  r'C:\Users\C3407841.C3407841CD\Desktop\cadi.jpg'
# img = cv2.imread(imgpath)
# img_rotate(img,-1.5059505383173625)


# img = cv2.imread('./datasets/clis1ok/1585365648208.jpg',cv2.IMREAD_GRAYSCALE)
# sobel = sobel_extract(img)
# cv2.imshow('img',img)
# cv2.imshow('sobel',sobel)
# cv2.waitKey()

# print(np.exp(0.00832500098))
# print(4e-3)
# print(-8.32500098e+03)
# print(np.exp(-8.32500098e+03))
