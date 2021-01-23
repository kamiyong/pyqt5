import torch
import torch.nn.functional as F
import cv2.cv2 as cv2
import numpy as np
import os, sys

sys.path.append('.')

from MODEL.zhang.img_process import img_process_p2, img_process_p1, find_anchor_point
from MODEL.zhang.train_simese import SimeseNet

img_h = 30
img_w = 30
input_shape = (1, img_h, img_w)
model_path1 = r'./MODEL/zhang/p1_epoch200lr0.0001loss0.02467060769413365.pth'
model_path2 = r'./MODEL/zhang/p2_epoch200lr0.0001loss0.023622587767022196.pth'
temp_img_path1 = r'./MODEL/zhang/p1_temp.jpg'
temp_img_path2 = r'./MODEL/zhang/p2_temp.jpg'
model1 = SimeseNet((1, 30, 30))
model1.load_state_dict(torch.load(model_path1)['model_state_dict'])
model1.eval()
model2 = SimeseNet((1, 50, 50))
model2.load_state_dict(torch.load(model_path2)['model_state_dict'])
model2.eval()


def generate_input_img(img):
    input_img = img.astype(np.float32) / 255.0
    height, width = img.shape
    input_img = np.reshape(input_img, (1, height, width, 1))
    input_img = torch.from_numpy(input_img).type(torch.FloatTensor).permute(0, 3, 1, 2)
    return input_img


temp_img1 = cv2.imread(temp_img_path1, cv2.IMREAD_GRAYSCALE)
temp_img1 = generate_input_img(temp_img1)
temp_img2 = cv2.imread(temp_img_path2, cv2.IMREAD_GRAYSCALE)
temp_img2 = generate_input_img(temp_img2)


def normalize_img(img):
    img = img.astype(np.float32) / 255.0
    return img


def val(img, model_path, localtion, type):
    def predict_distance(img_process, temp_img, model):
        inputimg = img_process(img, type, (anchor_x, anchor_y))
        # cv2.imshow('inputimg', inputimg)
        # cv2.waitKey()
        if inputimg is None: return -1
        # cv2.imshow('inputimg',inputimg)
        # cv2.waitKey()
        inputimg = generate_input_img(inputimg)
        # print(inputimg.size())
        out1, out2 = model(temp_img, inputimg)
        distance = F.pairwise_distance(out1, out2, keepdim=True)
        result = torch.where(distance.gt(0.5), torch.full_like(distance, -1), torch.full_like(distance, 1))
        return result.item()

    anchor_loc = find_anchor_point(img, 0, 0)
    # print(anchor_loc)
    if anchor_loc is None:
        anchor_x, anchor_y = 0, 0
    else:
        anchor_x, anchor_y = anchor_loc
    if localtion == 2:
        result = predict_distance(img_process_p2, temp_img2, model2)

    elif localtion == 1:
        result = predict_distance(img_process_p1, temp_img1, model1)
    else:
        raise ValueError('location must be integer 1 or 2')

    return result


if __name__ == "__main__":

    imgdir = r'C:\Users\C3408537\Desktop\20-12-31\OK'

    files = os.listdir(imgdir)
    import time

    for file in files:
        if file.endswith('jpg'):
            imgpath = os.path.join(imgdir, file)
            img = cv2.imread(imgpath)
            start = time.time()
            result1 = val(img, None, 1, 'B')
            result2 = val(img, None, 2, 'B')
            if result1 == -1 or result2 == -1:
                cv2.waitKey()
            print(result1, result1, imgpath, time.time() - start)
