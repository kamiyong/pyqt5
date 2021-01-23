import cv2.cv2 as cv2
import os
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from MODEL.zhang.data_enhance import dilated, eroded
import joblib
import numpy as np

'''140B 142B'''
# x,y,w,h
P1_LOCATION = (1 / 10, 11 / 45, 1 / 10, 2 / 15)  # (1/10,11/45,1/10,2/15) #(340,250,430,340)#(340,250,430,340)
P2_LOCATION = (5 / 8, 2 / 9, 1 / 10, 2 / 15)  # (5/8,2/9,1/10,2/15) #(1539,217,1629,307)
# 大盘的相对定位
P1_ANCHOR_RELATIVE_LOCATION_B = (-(1 / 80) - (1 / 20), -(37 / 100) - (1 / 15), 1 / 10, 2 / 15)
P2_ANCHOR_RELATIVE_LOCATION_B = ((81 / 160) - (1 / 20), -(17 / 45) - (1 / 15), 1 / 10, 2 / 15)
'''140s 142s'''
P1_LOCATION_S = (1 / 8, 14 / 45, 1 / 10, 2 / 15)
P2_LOCATION_S = (47 / 80, 13 / 45, 1 / 10, 2 / 15)

P1_ANCHOR_RELATIVE_LOCATION_S = (-(1 / 80) - (1 / 20), -(3 / 10) - (1 / 15), 1 / 10, 2 / 15)
P2_ANCHOR_RELATIVE_LOCATION_S = ((91 / 200) - (1 / 20), -(29 / 90) - (1 / 15), 1 / 10, 2 / 15)

model1 = joblib.load('./MODEL/zhang/p1.model')
model2 = joblib.load('./MODEL/zhang/p2.model')


def find_anchor_point(img, ww, hh):
    crop = img[:, :img.shape[1] // 2]
    crop = cv2.GaussianBlur(crop, (5, 5), 0)

    def get_white(src):
        # 白色
        lower_white = np.array([0, 0, 221])
        upper_white = np.array([180, 45, 255])
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        return mask

    wmask = get_white(crop)
    ero = eroded(wmask)
    cnts, _ = cv2.findContours(ero, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_cnts = []

    def cnt_area(cnt):
        area = cv2.contourArea(cnt)
        return area

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
            if y0 < y and x > 240 and x < 300:
                cnt0 = ncnt
                y0 = y
        new_cnts = cnt0

    def get_bound(cnts):
        allarea = np.concatenate(cnts)
        x, y, w, h = cv2.boundingRect(allarea)
        return x, y, w, h

    cut_img = img
    if len(new_cnts) > 0:
        # print(len(new_cnts))
        x1, y1, w1, h1 = get_bound(new_cnts)
        return x1, y1
        # cut_img = img[y1 + h1 - 25 - hh:y1 + h1 - hh, x1 + w1 - 52 - ww:x1 + w1 - ww]  # new
    # return cut_img
    return None


def img_process_p1(img, type, anchor_loc, outputsize=(30, 30)):
    height, width = img.shape[:2]
    anchor_x, anchor_y = anchor_loc
    # print(anchor_x,anchor_y)
    if type.find('B') != -1:
        if anchor_x == 0 and anchor_y == 0:
            src = img[int(height * P1_LOCATION[1]):int(height * (P1_LOCATION[1] + P1_LOCATION[3])),
                  int(width * P1_LOCATION[0]):int(width * (P1_LOCATION[0] + P1_LOCATION[2]))]
        else:
            src = img[int(anchor_y + height * P1_ANCHOR_RELATIVE_LOCATION_B[1]):int(
                anchor_y + height * P1_ANCHOR_RELATIVE_LOCATION_B[1] + height * P1_ANCHOR_RELATIVE_LOCATION_B[3]),
                  int(anchor_x + width * P1_ANCHOR_RELATIVE_LOCATION_B[0]):int(
                      anchor_x + width * P1_ANCHOR_RELATIVE_LOCATION_B[0] + width * P1_ANCHOR_RELATIVE_LOCATION_B[2])]
    elif type.find('S') != -1:
        if anchor_x == 0 and anchor_y == 0:
            src = img[int(height * P1_LOCATION_S[1]):int(height * (P1_LOCATION_S[1] + P1_LOCATION_S[3])),
                  int(width * P1_LOCATION_S[0]):int(width * (P1_LOCATION_S[0] + P1_LOCATION_S[2]))]
        else:
            src = img[int(anchor_y + height * P1_ANCHOR_RELATIVE_LOCATION_S[1]):int(
                anchor_y + height * P1_ANCHOR_RELATIVE_LOCATION_S[1] + height * P1_ANCHOR_RELATIVE_LOCATION_S[3]),
                  int(anchor_x + width * P1_ANCHOR_RELATIVE_LOCATION_S[0]):int(
                      anchor_x + width * P1_ANCHOR_RELATIVE_LOCATION_S[0] + width * P1_ANCHOR_RELATIVE_LOCATION_S[2])]
    else:
        raise ValueError('parameter \'type\' must be in [\'B\' ,\'S\'] or [\'140B\',\'140S\',\'142B\',\'142S\']')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    ret, thres_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    erode_img = eroded(thres_img, (3, 3))
    dilate_img = dilated(erode_img, (3, 3))
    # blur = cv2.GaussianBlur(gray,(17,17),0)
    canny = cv2.Canny(dilate_img, 130, 150, apertureSize=7, )
    # cv2.imshow('dilate_img1',dilate_img)
    # cv2.imshow('canny1',canny)

    # cv2.waitKey()
    # 霍夫圆检测
    circles = cv2.HoughCircles(thres_img, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=5, param1=200, param2=16, minRadius=20, maxRadius=30)
    # print(circles)
    if circles is not None:

        if circles.shape[1] > 1:
            circle = circles[0][np.argmax(circles[0, :, 2])]
            # print(circles)
        else:
            circle = circles[0, 0, :]
        box_radius = 25
        circle_box = None
        for circle in circles[0]:
            cent_x, cent_y, _ = circle
            if cent_x > 3 * width / 16 and cent_x < 13 * width / 16 \
                    and cent_y > 25 * height / height and cent_y < 95 * height / height:
                # 获取包含螺丝的一个大概范围
                circle_box = gray[int(cent_y - box_radius):int(cent_y + box_radius),
                             int(cent_x - box_radius):int(cent_x + box_radius)]
        if circle_box is not None:
            # print(circles)
            ret, thres_img2 = cv2.threshold(circle_box, 100, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thres_img2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            length_rang = (30, 40)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                if w > length_rang[0] and w < length_rang[1] and \
                        h > length_rang[0] and h < length_rang[1]:  # and cv2.contourArea(contour) > w*h/2
                    # print(x,y,w,h)
                    result_img = circle_box[y:y + h, x:x + w]
                    result_img = cv2.resize(result_img, outputsize)

                    return result_img

            # cv2.imshow('thres_img2',thres_img2)
            # cv2.imshow('circle_box',circle_box)
            # cv2.waitKey()

    # else:
    #     cv2.imshow('src',src)
    #     cv2.imshow('canny',canny)
    #     cv2.imshow('thres_img',thres_img)
    #     cv2.waitKey()


def data_augment():
    srcimgpath = r'E:\C3407841\task\AWFeedbar\datasets\image\NG\NoScrew\H4HDP47ZQ07Y_20-11-09_18-01-20.jpg'

    luosi_location_2 = (1045, 256, 50, 40)
    srcimg = cv2.imread(srcimgpath)
    cropimg = srcimg[luosi_location_2[1]:luosi_location_2[1] + luosi_location_2[3],
              luosi_location_2[0]:luosi_location_2[0] + luosi_location_2[2]]
    # cv2.imshow('crio',cropimg)
    # cv2.waitKey()
    height, width = cropimg.shape[:2]
    stride = (2, 2)
    kernel = (46, 34)

    result_size = (50, 50)
    curx, cury = kernel
    img_dir, img_name = os.path.split(srcimgpath)
    filename, suffix = os.path.splitext(img_name)
    count = 1

    def save_img(img):

        if result_size is not None:
            result_img = cv2.resize(img, result_size)
            cv2.imwrite(os.path.join(img_dir, filename + '-' + str(count) + suffix), result_img)

    x_rang = int((width - kernel[0]) / 2 + 1)
    y_rang = int((height - kernel[1]) / 2 + 1)
    count = 1
    print(x_rang, y_rang)
    for y in range(y_rang):
        for x in range(x_rang):
            cur_img = cropimg[cury - kernel[1]:cury, curx - kernel[0]:curx]
            save_img(cur_img)
            curx += stride[0]
            count += 1
        cury += stride[1]
        # while curx < width or cury < height:
        #     if curx < width and cury < height:
        #         cur_img = cropimg[cury-kernel[1]:cury,curx-kernel[0]:curx]
        #         save_img(cur_img)
        #         curx+=stride[0]
        #         cury+=stride[1]
        #     elif curx < width:
        #         cur_img = cropimg[height-kernel[1]:height,curx-kernel[0]:curx]
        #         save_img(cur_img)
        #         curx+=stride[0]
        #     elif cury < height :
        #         cur_img = cropimg[cury-kernel[1]:cury,width-kernel[0]:width]
        #         save_img(cur_img)
        #         cury+=stride[1]
        count += 1


def bright_adjust(gray_img):
    '''
        调整亮度，使得明暗分明
    '''
    alpha = 2
    sub_img = alpha * np.log2(2 + gray_img)
    result_img = gray_img - sub_img
    result_img[result_img < 0] = 0

    return result_img.astype(np.uint8)


def img_process_p2(img, type, anchor_loc, outputsize=(50, 50)):
    height, width = img.shape[:2]
    anchor_x, anchor_y = anchor_loc
    # print(anchor_x,anchor_y)
    if type.find('B') != -1:
        if anchor_x == 0 and anchor_y == 0:
            src = img[int(height * P2_LOCATION[1]):int(height * (P2_LOCATION[1] + P2_LOCATION[3])),
                  int(width * P2_LOCATION[0]):int(width * (P2_LOCATION[0] + P2_LOCATION[2]))]
        else:
            src = img[int(anchor_y + height * P2_ANCHOR_RELATIVE_LOCATION_B[1]):int(
                anchor_y + height * P2_ANCHOR_RELATIVE_LOCATION_B[1] + height * P2_ANCHOR_RELATIVE_LOCATION_B[3]),
                  int(anchor_x + width * P2_ANCHOR_RELATIVE_LOCATION_B[0]):int(
                      anchor_x + width * P2_ANCHOR_RELATIVE_LOCATION_B[0] + width * P2_ANCHOR_RELATIVE_LOCATION_B[2])]
    elif type.find('S') != -1:
        if anchor_x == 0 and anchor_y == 0:
            src = img[int(height * P2_LOCATION_S[1]):int(height * (P2_LOCATION_S[1] + P2_LOCATION_S[3])),
                  int(width * P2_LOCATION_S[0]):int(width * (P2_LOCATION_S[0] + P2_LOCATION_S[2]))]
        else:
            src = img[int(anchor_y + height * P2_ANCHOR_RELATIVE_LOCATION_S[1]):int(
                anchor_y + height * P2_ANCHOR_RELATIVE_LOCATION_S[1] + height * P2_ANCHOR_RELATIVE_LOCATION_S[3]),
                  int(anchor_x + width * P2_ANCHOR_RELATIVE_LOCATION_S[0]):int(
                      anchor_x + width * P2_ANCHOR_RELATIVE_LOCATION_S[0] + width * P2_ANCHOR_RELATIVE_LOCATION_S[2])]

    else:
        raise ValueError('parameter \'type\' must be in [\'B\' ,\'S\'] or [\'140B\',\'140S\',\'142B\',\'142S\']')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    # cv2.imshow('gray2',gray)
    # hist = cv2.equalizeHist(gray)
    # cv2.imshow('hist2',hist)
    ret, thres_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    canny = cv2.Canny(blur, 100, 150, apertureSize=7)
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, dp=1,
                               minDist=6, param1=300, param2=20, minRadius=23, maxRadius=28)
    # print(circles)
    # cv2.imshow('canny2',canny)
    # cv2.imshow('thres_img2',thres_img)
    # cv2.waitKey()

    if circles is not None:
        circles = np.mean(circles, axis=1, keepdims=True)
        for circle in circles[0, :]:
            cent_x, cent_y, radius = circle
            # print(circle)
            # if cent_x < 0.4*width or cent_x > 0.6*width \
            #     or cent_y < 0.4*height or cent_y > 0.6*height:
            #     continue

            circle_img = gray[int(cent_y - radius):int(cent_y + radius), int(cent_x - radius):int(cent_x + radius)]
            # print(circle)
            # cv2.circle(img,(cent_x,cent_y),int(radius),(0,255,0),3)
            # cv2.imshow('img',img)
            # cv2.waitKey()
            # cv2.imshow('circle_img',circle_img)
            result_img = cv2.resize(circle_img, outputsize)
            return result_img


def batch_process(imgdir, save_dir=None):
    assert os.path.exists(imgdir), 'cannot found directory ' % imgdir
    datasets = []
    img_names = []
    train_count = 0
    for root, dirs, files in os.walk(imgdir):
        for file in files:
            if file.endswith('.jpg'):
                filepath = os.path.join(root, file)
                obj_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                # root_dir,dirname = os.path.split(root)
                # obj_img = img_process_p1(img,dirname)
                if obj_img is None:
                    print(filepath)
                    continue
                if save_dir is not None:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    cv2.imwrite(os.path.join(save_dir, file), obj_img)
                datasets.append(obj_img)
                img_names.append(filepath)
                train_count += 1
    print('train num ', train_count)
    return datasets, img_names


def batch_crop():
    img_dir = r'E:\C3407841\task\AWFeedbar\Feedbar_10.26\NG MODEL\140B\p2'
    save_dir = r'E:\C3407841\task\AWFeedbar\datasets\1026\ng'
    # crop_box1 = (312,201,402,291)#(287,183,377,273)#(332,427,422,517)#(413,334,503,424)#(340,250,430,340)
    # crop_box2 = (1545,167,1645,267)#(1571,141,1671,241)#(1651,419,1751,519)#(1479,306,1569,396)#(1539,217,1629,307)
    # crop_box1 = (308,200,398,290)
    crop_box2 = (1558, 206, 1658, 306)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    files = os.listdir(img_dir)
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(img_dir, file))
            # print(img.shape)
            # img1 = img[crop_box1[1]:crop_box1[3],crop_box1[0]:crop_box1[2]]
            img2 = img[crop_box2[1]:crop_box2[3], crop_box2[0]:crop_box2[2]]
            # img1 = cv2.resize(img1,(90,90))
            img2 = cv2.resize(img2, (90, 90))
            # cv2.imwrite(os.path.join(save_dir,'p1'+file),img1)
            cv2.imwrite(os.path.join(save_dir, 'p2' + file), img2)


def train_p2(datasets, model_path):
    '''
    datasets:数据集
    model_path:模型存储路径
    n_neighbours:lof的局部密度阈值
    n_components:pca的降维后的特征维度,当pca=True时生效
    pca:是否使用pca降维
    '''
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True)  # 35
    # ocs = OneClassSVM(kernel = 'rbf')
    # IF = IsolationForest(n_estimators=30)
    results = lof.fit(datasets)
    results = lof.predict(datasets)
    print(np.sum(results > 0) / datasets.shape[0])

    joblib.dump(lof, model_path)
    return results


def train_p1(datasets, model_path):
    '''
    datasets:数据集
    model_path:模型存储路径
    n_neighbours:lof的局部密度阈值
    n_components:pca的降维后的特征维度,当pca=True时生效
    pca:是否使用pca降维
    '''

    lof = LocalOutlierFactor(n_neighbors=60, novelty=True)  # 20
    # ocs = OneClassSVM(kernel = 'rbf')
    # IF = IsolationForest(n_estimators=30)
    results = lof.fit(datasets)
    results = lof.predict(datasets)
    print(np.sum(results > 0) / datasets.shape[0])

    joblib.dump(lof, model_path)
    return results


def batch_val(model_path, datasets):
    model = joblib.load(model_path)
    results = model.predict(datasets)
    return results


def val(img, model_path, location, type):
    '''
    @param img:8-bit img
    @param model_path:模型路径
    @param location screw location .must be 1 or 2 
    '''
    anchor_loc = find_anchor_point(img, 0, 0)
    # print(img.shape)
    if anchor_loc is None:
        anchor_x, anchor_y = 0, 0
    else:
        anchor_x, anchor_y = anchor_loc

    # print(anchor_x,anchor_y)
    def predict(input_img, model):
        data = input_img.reshape((1, -1))
        result = model.predict(data)
        return result

    if location == 1:
        obj_img = img_process_p1(img, type, (anchor_x, anchor_y))
        if obj_img is None: return -1
        result = predict(obj_img, model1)
    elif location == 2:
        obj_img = img_process_p2(img, type, (anchor_x, anchor_y))
        if obj_img is None: return -1
        result = predict(obj_img, model2)
    else:
        raise ValueError('location must be 1 or 2 ')

    # data = obj_img.reshape((1,-1))
    # model = joblib.load(model_path)
    # result = model.predict(data)
    return result[0]

# if __name__ == "__main__":
# imgdir =r'E:\C3407841\task\AWFeedbar\datasets\image\p1'
# model_path=r'./p1_v2.model'
# datasets,imgnames = batch_process(imgdir,None)
# datasets = np.array(datasets).reshape((len(datasets),-1))
# train_p1(datasets,model_path=r'./p1_v2.model')
# results = batch_val(model_path,datasets)

# #print(results)
# for index,res in enumerate(results):
#     if res == -1:
#         print(imgnames[index])

# #save_dir = r'E:\C3407841\task\AWFeedbar\datasets\image\p1'
# datasets1,imgnames1 = batch_process(imgdir,None)
# datasets1 = np.array(datasets1).reshape((len(datasets1),-1))/255.0
# imgdir = r'E:\C3407841\task\AWFeedbar\datasets\image\p2\ok'
# #save_dir = r'E:\C3407841\task\AWFeedbar\datasets\image\p1'
# datasets2,imgnames2 = batch_process(imgdir,None)
# datasets2 = np.array(datasets2).reshape((len(datasets2),-1))/255.0
# from sklearn.decomposition import PCA
# from matplotlib import pyplot as plt
# pca = PCA(n_components=2)
# result_d1 = pca.fit_transform(datasets1)
# result_d2 = pca.fit_transform(datasets2)
# plt.scatter(result_d1[:,0],result_d1[:,1])
# plt.scatter(result_d2[:,0],result_d2[:,1])
# plt.legend()
# plt.show()

# #results = train_p2(datasets,20,'./p1_1.model')
# results = batch_val('./p1_1.model',datasets)
# #print(results)
# for index,res in enumerate(results):
#     if res == -1:
#         print(imgnames[index])

# imgpath = r'E:\C3407841\task\AWFeedbar\datasets\image\src\140B\H4HDP2FUQ07Y_20-11-09_17-53-12.jpg'
# #imgpath = r'E:\C3407841\task\AWFeedbar\datasets\train\ng\p120201020-161552-604.jpg'
# img = cv2.imread(imgpath)
# img_process_p1(img,'B')
# cv2.waitKey()

# model_path1 = r'./p1.model'
# model_path2 = r'./p2.model'
# imgdir = r'E:\C3407841\task\AWFeedbar\test\2020-11-27-testing-result\NG'
# for root,dirs,files in os.walk(imgdir):
# for file in files:
# if file.endswith('.jpg'):
# imgpath = os.path.join(root,file)
# img = cv2.imread(imgpath)
# result1 = val(img,model_path1,1,'B')
# result2 = val(img,model_path2,2,'B')
# print(imgpath,result1,result2)
# if result1 == -1 or result2 == -1:
# #cv2.waitKey()

# imgpath = r'D:\soft\Test\MODEL\zhang\07YH4HDP5UBQ07Y_20-11-10_16-51-47.jpg'

# #r'E:\C3407841\task\AWFeedbar\Feedbar_10.26\140b\20201026-150901-744.jpg'
# model_path = r'./p2.model'
# img = cv2.imread(imgpath)
# print(val(img,model_path,2,'B'))

# cv2.waitKey()
# H4HDQ07VQ123_20-11-18_10-19-36.jpg
# E:\C3407841\task\AWFeedbar\datasets\image\src\140S\H4HDP25XQ07T_20-11-10_17-22-44.jpg
# E:\C3407841\task\AWFeedbar\datasets\image\src\140S\H4HDP59CQ07T_20-11-11_16-11-11.jpg

# E:\C3407841\task\AWFeedbar\datasets\image\src\142S\H4HDQ05VQ123_20-11-18_10-10-12.jpg
