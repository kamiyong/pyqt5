import torch
import torchvision
import torch.nn.functional as F
from torch.nn import Module, BatchNorm3d, BatchNorm2d, ReLU6, Linear, AvgPool3d, Conv3d, Conv2d, AvgPool2d, ReLU, \
    MaxPool2d, Dropout2d, Softmax, Sequential
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2.cv2 as cv2

mean = 0
std = 1
batch_size = 16
epoch = 200
lr = 1e-4
class_num = 2
img_h = 30
img_w = 30
input_shape = (1, img_h, img_w)
pretrain = False
premodelpath = r'E:\C3407841\VSCode-workspace\APPTVAOI\aw_model\p1_epoch200lr0.0001loss0.02467060769413365.pth'
model_dir = './aw_model_p1/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dir = r'E:\C3407841\task\AWFeedbar\datasets\image\p1_1'


def generate_datasets(datasets_dir, data_num=200, main_cat='ok', normalize=False, ):
    def load_datasets(datasets_dir):
        datasets = []
        names = []
        label_map = {}
        label_index = 0
        for root, dirs, files in os.walk(datasets_dir):
            file_suffix = ['.jpg', '.jpeg', '.JPG', '.JPEG']
            for file in files:
                filename, suffix = os.path.splitext(file)
                if suffix in file_suffix:
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    _, label = os.path.split(root)
                    datasets.append(img)
                    names.append(label)
                    if label not in label_map.keys():
                        print(label)
                        label_map[label] = label_index
                        label_index += 1
        return datasets, names, label_map

    datasets, labels, label_map = load_datasets(datasets_dir)
    datasets_len = len(datasets)
    data_pair = []
    label_pair = []
    main_label = label_map[main_cat]
    print('main_label is ', main_label)

    while True:
        indexs = np.random.randint(datasets_len, size=2)
        label1 = label_map[labels[indexs[0]]]
        img1 = datasets[indexs[0]]
        label2 = label_map[labels[indexs[1]]]
        img2 = datasets[indexs[1]]

        if label1 == label2 and label2 == main_label:
            label_pair.append(1)
            data_pair.append([img1, img2])
        elif label1 == main_label:
            label_pair.append(0)
            data_pair.append([img1, img2])
        elif label2 == main_label:
            label_pair.append(0)
            data_pair.append([img2, img1])
        if len(data_pair) == data_num: break
    return data_pair, label_pair


def weight_init(m):
    modulename = m.__class__.__name__

    if modulename.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif modulename.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)
    elif modulename.find('Linear') != -1:
        m.weight.data.normal_(0, 0.1)
        m.bias.data.zero_()


class DatasetGenerator(Dataset):
    def __init__(self, dataset_dir, main_cat, img_process=None, args=None):
        '''
            @param dataset_dir : 类别根目录
            @param main_cat:主类别,除此以外的类别都为ng,
            @param img_process :图像处理方法
            @param args :图像处理方法的参数
        '''
        super(DatasetGenerator, self).__init__()
        self.img_process = img_process
        self.args = args
        self.data_paths, self.labels, self.label_map = self._load_datasets(dataset_dir)
        self.main_label = main_cat  # self.label_map[main_cat]
        self.main_cat = main_cat
        self.data_num = len(self.labels)

    def _load_datasets(self, datasets_dir):
        data_paths = []
        names = []
        label_map = {}
        label_index = 0
        file_suffix = ['.jpg', '.jpeg', '.JPG', '.JPEG']
        for root, dirs, files in os.walk(datasets_dir):
            for file in files:
                filename, suffix = os.path.splitext(file)

                if suffix in file_suffix:

                    _, label = os.path.split(root)
                    data_paths.append(os.path.join(root, file))
                    names.append(label)
                    if label not in label_map.keys():
                        label_map[label] = label_index
                        label_index += 1
        return data_paths, names, label_map

    def __len__(self):
        return self.data_num

    def preprocess(self, image):
        image = image.astype(np.float32) / 255.0
        res = (image - mean) / std
        return res

    def __getitem__(self, index):
        while True:
            indexs = np.random.randint(self.data_num, size=2)
            label1 = self.labels[indexs[0]]  # self.label_map[self.labels[indexs[0]]]
            label2 = self.labels[indexs[1]]  # self.label_map[self.labels[indexs[1]]]
            # if label1 == label2 and label1 != self.main_label:
            #     continue
            img1 = cv2.imread(self.data_paths[indexs[0]], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.data_paths[indexs[1]], cv2.IMREAD_GRAYSCALE)
            # print(label1==label2 and label1==self.main_label)
            # cv2.imshow('img1',img1)
            # cv2.imshow('img2',img2)

            # cv2.waitKey()
            if callable(self.img_process):
                img1 = self.img_process(img1, self.args)
                img2 = self.img_process(img2, self.args)

            if len(img1.shape) == 2:
                img1 = np.expand_dims(img1, 2)
            if len(img2.shape) == 2:
                img2 = np.expand_dims(img2, 2)
            img1 = self.preprocess(img1)
            img2 = self.preprocess(img2)
            img1 = torch.from_numpy(img1).type(torch.FloatTensor).permute(2, 0, 1)
            img2 = torch.from_numpy(img2).type(torch.FloatTensor).permute(2, 0, 1)
            if label1 == label2 and label2 == self.main_label:
                return img1, img2, 0
            elif label1 == self.main_label:
                return img1, img2, 1
            elif label2 == self.main_label:
                return img2, img1, 1
            elif label1 == label2 and label1 != self.main_label:
                return img1, img2, 0


class Conv3_BN(Module):

    def __init__(self, in_chn, out_chn, k_size, stride=1, padding=0):
        super(Conv3_BN, self).__init__()
        self.conv = Conv3d(in_channels=in_chn, out_channels=out_chn,
                           kernel_size=k_size, stride=stride, padding=padding)
        self.bn = BatchNorm3d(out_chn)
        self.relu = ReLU6(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.relu(out)
        # out = self.bn(out)
        return out


class Conv_BN_RELU(Module):
    def __init__(self, in_chn, out_chn, k_size, stride=1, padding=0):
        super(Conv_BN_RELU, self).__init__()
        self.conv = Conv2d(in_channels=in_chn, out_channels=out_chn,
                           kernel_size=k_size, stride=stride, padding=padding)
        self.bn = BatchNorm2d(out_chn)
        self.relu = ReLU(inplace=True)

    def forward(self, input):
        out = self.conv(input)
        out = self.relu(out)
        # out = self.bn(out)
        return out


class SimeseNet(Module):
    def __init__(self, input_shape):
        '''
        @param input_shape:(b,c,h,w)
        '''
        super(SimeseNet, self).__init__()
        self.input_shape = input_shape

        self.cnn = torch.nn.Sequential(
            Conv_BN_RELU(input_shape[0], 32, 3, 2, 1),  # (b,c,46,46)
            # self.drop1 = Dropout2d(.2)
            Conv_BN_RELU(32, 32, 3, 2, 1),
            MaxPool2d((2, 2), 1),
            # self.drop2 = Dropout2d(.2)
            Conv_BN_RELU(32, 64, 3, 2, 1),
            # self.drop3 = Dropout2d(.2)
            Conv_BN_RELU(64, 64, 3, 2, 1),
            # self.drop4 = Dropout2d(.2)
            MaxPool2d((2, 2), 1)
        )
        if input_shape[1] == 50:
            linear1 = Linear(in_features=64 * 2 * 2, out_features=512)
        elif input_shape[1] == 30:
            linear1 = Linear(in_features=64, out_features=512)
        self.fc = Sequential(
            linear1,
            ReLU(inplace=True),
            Dropout2d(.4),
            Linear(in_features=512, out_features=2)
        )

    def _forward(self, inp):

        out = self.cnn(inp)
        b, c, height, width, = out.size()

        out = out.view((b, -1))
        out = self.fc(out)

        return out

    def forward(self, input1, input2):
        # print(input.size())
        # print(input.size())

        out1 = self._forward(input1)
        out2 = self._forward(input2)
        return out1, out2


# 自定义损失函数
class ContrastiveLoss(Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, input1, input2, labels):
        '''
        param input1 and input2 should be in shape of (b,d)
        '''

        distance = F.pairwise_distance(input1, input2, keepdim=True)  # F.pairwise_distance(input1,input2,keepdim=True)
        # print(distance.size())
        # print(labels.size())
        # print(labels.float()*torch.pow(torch.clamp(2-distance,min=0.0),2))
        ng_loss = labels.float() * torch.pow(torch.clamp(1 - distance, min=0.0), 2)
        # print(ng_loss.flatten())
        loss = torch.mean(((1 - labels.float()) * torch.pow(distance, 2)) + ng_loss)

        return loss


# 修正梯度爆炸
def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0


def move(dataset_loader, net, criterion, optimizer, train=True):
    net = net.to(device)
    criterion = criterion.to(device)
    if train:
        net.train()
    epoch_loss = 0
    for batch_index, (img1, img2, labels) in enumerate(dataset_loader):
        img1, img2 = img1.to(device), img2.to(device)
        labels = labels.view((labels.size()[0], 1)).to(device)
        out1, out2 = net(img1, img2)
        # print(labels)
        # print(imgs.size())
        loss = criterion(out1, out2, labels)
        net.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def val(model_path):
    dataset = DatasetGenerator(train_dir, main_cat='ok', )
    # 数据加载器
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 定义网络
    net = SimeseNet(input_shape, )
    net = net.to(device)
    net.load_state_dict(torch.load(model_path)['model_state_dict'])
    net.eval()
    acc = 0
    val_len = 0
    for batch_index, (img1, img2, labels) in enumerate(val_loader):
        img1, img2 = img1.to(device), img2.to(device)
        labels = labels.view((labels.size()[0], 1)).to(device)
        out1, out2 = net(img1, img2)
        # print(out1,out2)
        # distance_square = torch.sum(torch.pow(out1-out2,2),dim=1)
        # print(distance_square)
        distance = F.pairwise_distance(out1, out2, keepdim=True)
        # distance = distance_square.sqrt()

        # print(distance.view(1,-1))
        # print(labels.view((1,-1)))
        # print(distance.flatten())
        predic = distance.gt(0.5).long()
        # print(labels.flatten())
        # print(predic.flatten())
        acc += (labels == predic).sum().item()
        val_len += labels.size(0)
    # print(acc)
    # print(val_len)
    return acc / val_len


def inference(dataset_loader, net, criterion, optimizer):
    import math
    loss_history = []
    cur_loss = 0
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if pretrain:
        ckpt = torch.load(premodelpath)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optim_state_dict'])
    for i in range(epoch):
        epoch_loss = move(dataset_loader, net, criterion, optimizer, train=True)
        loss_history.append(epoch_loss)
        model_path = os.path.join(model_dir, 'epoch{}lr{}loss{}.pth'.format(epoch, lr, epoch_loss))
        if cur_loss == 0: cur_loss = epoch_loss + 0.1
        if epoch_loss < cur_loss:
            cur_loss = epoch_loss
            torch.save({
                'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict()
            }, model_path)
            acc = val(model_path)
            print(f'################## epoch:{i},val_acc:{acc},loss:{epoch_loss} #################################')


def train():
    dataset = DatasetGenerator(train_dir, main_cat='ok', )
    # 数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 定义网络
    net = SimeseNet(input_shape, )
    net.register_backward_hook(backward_hook)
    # 损失函数
    criterion = ContrastiveLoss()
    # 优化器
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)

    # net.apply(weight_init)
    inference(train_loader, net, criterion, optimizer)

# train()

# datasets_dir = r'E:\C3407841\task\APPLE_AOI\datasets\73-IC-Linear\simese'
# datasets,labels  = generate_datasets(datasets_dir)
# print(labels)
