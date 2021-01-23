import torch
import torchvision
import torch.nn.functional as F
from torch.nn import Module,BatchNorm3d,BatchNorm2d,ReLU6,Linear,AvgPool3d,Conv3d,Conv2d,AvgPool2d,ReLU,MaxPool2d,Dropout2d,Softmax,Sequential
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset ,DataLoader
import numpy as np
import os
import cv2.cv2 as cv2



def eroded(src, size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=size)
    erode = cv2.erode(src, kernel)
    return erode


def dilated(src, size=(3, 3)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=size)
    dilate = cv2.dilate(src, kernel)
    return dilate  


def get_roi_data_balde(image_path, watch_size='B', target_size=(50, 50)):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = img.shape[0], img.shape[1]
    roi_zero = blur[100:height, 0:width//4]
    ROI_zeros = img[100:height, 0:width//4]
    ret, thresh = cv2.threshold(roi_zero, 180, 255,cv2.THRESH_BINARY)
    erode = eroded(thresh, (9, 9))
    dilate = dilated(erode, (7, 7))
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    roi_contour = []
    if watch_size =='S': 
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 30 < h < 100 and 35 < w < 80  and 580 > y > 500:
                roi_contour.append(contour)

          
    elif watch_size == 'B':  
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 40 < h < 120 and 30 < w < 85  and 580 > y > 480:
                roi_contour.append(contour)
        
    else:
        print('parameter error')
        # sys.exit(0)
        


    if len(roi_contour) != 1:
        return 0
    else:
        if watch_size =='S': 
            new_x, new_y = 0, 0
            for i in range(roi_contour[0].shape[0]):
                x, y = roi_contour[0][i][0][0], roi_contour[0][i][0][1]
                if x > new_x:
                    new_x = x
                if y > new_y:
                    new_y = y


            if  330 < new_x < 380  and 570 < new_y < 620:

                final_roi = img[new_y-280+100:new_y-200+100,new_x-100:new_x-30]

            else:
                final_roi = img[420:500, 260:330]

        elif watch_size == 'B': 
            new_x, new_y = 0, 0
            for i in range(roi_contour[0].shape[0]):
                x, y = roi_contour[0][i][0][0], roi_contour[0][i][0][1]
                if x > new_x:
                    new_x = x
                if y > new_y:
                    new_y = y

            if  290 < new_x < 340  and 590 < new_y < 640:
                final_roi = img[new_y-345+100:new_y-265+100,new_x-100:new_x-30]

            else:
                final_roi = img[360:430, 215:290]
               

    final_roi = cv2.resize(final_roi, target_size)

    return final_roi


class SimeseNet(Module):
    def __init__(self,input_shape):
        '''
        @param input_shape:(b,c,h,w)
        '''
        super(SimeseNet,self).__init__()
        self.input_shape = input_shape

        self.cnn = torch.nn.Sequential(
            Conv_BN_RELU(input_shape[0],32,3,2,1),#(b,c,46,46)
            #self.drop1 = Dropout2d(.2)
            Conv_BN_RELU(32,32,3,2,1),
            MaxPool2d((2,2),1),
            #self.drop2 = Dropout2d(.2)
            Conv_BN_RELU(32,64,3,2,1),
            #self.drop3 = Dropout2d(.2)
            Conv_BN_RELU(64,64,3,2,1),
            #self.drop4 = Dropout2d(.2)
            MaxPool2d((2,2),1)
        )
        if input_shape[1] == 50:
            linear1 = Linear(in_features=64*2*2,out_features=512)
        elif input_shape[1] == 30:
            linear1 = Linear(in_features=64,out_features=512)
        self.fc = Sequential(
            linear1,
            ReLU(inplace=True),
            Dropout2d(.4),
            Linear(in_features=512,out_features=2)
        )
    def _forward(self,inp):
    
        out = self.cnn(inp)
        b,c,height,width, = out.size()

        out = out.view((b,-1))
        out = self.fc(out)
       
        return out
    def forward(self,input1,input2):
        #print(input.size())
        #print(input.size())
       
        out1 = self._forward(input1)
        out2 = self._forward(input2)
        return out1,out2


class Conv_BN_RELU(Module):
    def __init__(self,in_chn,out_chn,k_size,stride=1,padding=0):
        super(Conv_BN_RELU,self).__init__()
        self.conv = Conv2d(in_channels=in_chn,out_channels=out_chn,
            kernel_size=k_size,stride=stride,padding=padding)
        self.bn = BatchNorm2d(out_chn)
        self.relu = ReLU(inplace=True)
        
    def forward(self,input):
        out = self.conv(input)
        out = self.relu(out)
        #out = self.bn(out)
        return out


def generate_input_img(img):
    input_img = img.astype(np.float32)/255.0
    height,width = img.shape[:2]
    input_img = np.reshape(input_img,(1,height,width,3))
    input_img =torch.from_numpy(input_img).type(torch.FloatTensor).permute(0,3,1,2)
    return input_img
    

def model_predict_blade(final_roi_data, model_path, template_img_path):
    if isinstance(final_roi_data, int):
        return 0
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        test_img = generate_input_img(final_roi_data)
        
        # temp_img = cv2.imread(template_img_path,cv2.IMREAD_GRAYSCALE) 
        temp_img = cv2.imread(template_img_path) 
        temp_img = cv2.resize(temp_img,(50,50))
        temp_img = generate_input_img(temp_img)
        
        model = SimeseNet((3,50,50))
        # model.to(device)
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        model.eval()

        out1,out2 = model(temp_img,test_img)
        distance = F.pairwise_distance(out1,out2,keepdim=True)
        result = torch.where(distance.gt(0.5), torch.full_like(distance, -1), torch.full_like(distance, 1))
        return result.item()
    

if __name__ == '__main__':
    template_img_path = r'template_blade.jpg'
    model_path = r'./blade_rgb.pth'
    watch_size='S'

    image_path = r'E:\AW\mycodev2\train_siamese\mydata\S_OK_data\H4HF33Y7Q07V_21-01-16_16-17-15.jpg'
    final_roi_data = get_roi_data_balde(image_path, watch_size, target_size=(50, 50))
    res = model_predict_blade(final_roi_data, model_path, template_img_path)
    print(res) 
