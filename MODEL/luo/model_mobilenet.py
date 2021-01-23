from torch import nn
import torch
from torchvision import models


class MainModel(nn.Module):
    def __init__(self, **kwargs):
        super(MainModel, self).__init__()
        self.use_dcl = True
        self.num_classes = 3
        self.backone_arch = 'mobilenet_v2'
        self.use_Asoftmax = False

        self.model = getattr(models, self.backone_arch)()
        self.model = nn.Sequential(*list(self.model.children()))[:-1]
        self.classifier = nn.Linear(1280, self.num_classes, bias=False)
        self.dropout = nn.Dropout(0.2)

        if self.use_dcl:
            self.classifier_swap = nn.Linear(1280, 2 * self.num_classes, bias=False)
            self.Convmask = nn.Conv2d(1280, 1, 1, stride=1, padding=0, bias=True)
            self.avgpool2 = nn.AvgPool2d(2, stride=2)

    def forward(self, x, last_cont=None):
        x = self.model(x)
        if self.use_dcl:
            mask = self.Convmask(x)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)

        x = x.mean([2, 3])
        x = self.dropout(x)
        out = []
        out.append(self.classifier(x))

        if self.use_dcl:
            out.append(self.classifier_swap(x))
            out.append(mask)

        return out
