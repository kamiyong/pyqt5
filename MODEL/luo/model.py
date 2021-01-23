from torch import nn
import torch
from torchvision import models


class MainModel(nn.Module):
    def __init__(self, **kwargs):
        super(MainModel, self).__init__()
        self.use_dcl = True
        self.num_classes = 3
        self.backone_arch = 'resnet50'
        self.use_Asoftmax = False

        self.model = getattr(models, self.backone_arch)()
        self.model = nn.Sequential(*list(self.model.children()))[:-2]
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        if self.use_dcl:
            self.classifier_swap = nn.Linear(2048, 2*self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(2048, 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)


    def forward(self, x, last_cont=None):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        if self.use_Asoftmax:
            if last_cont is None:
                x_size = x.size(0)
                out.append(self.Aclassifier(x[0:x_size:2]))
            else:
                last_x = self.model(last_cont)
                last_x = self.avgpool(last_x)
                last_x = last_x.view(last_x.size(0), -1)
                out.append(self.Aclassifier(last_x))

        return out
