import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from base.base_net import BaseNet

class VISDA17ResNet(BaseNet):

    def __init__(self):
        super().__init__()
       
        self.backbone = ResNet50Fc()
        self.backbone = torch.nn.DataParallel(self.backbone, device_ids = [1,0])
        self.backbone.to("cuda:1")
        self.classifier1 = nn.Linear(2048, 12)
        self.classifier1 = torch.nn.DataParallel(self.classifier1, device_ids = [1,0])
        self.classifier1.to("cuda:1")
    def forward(self, x):
        feature = self.backbone(x)
        out1 = self.classifier1(feature)
        return  feature,out1

class ResNet50Fc(torch.nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).reshape(-1,2048)
        return x


