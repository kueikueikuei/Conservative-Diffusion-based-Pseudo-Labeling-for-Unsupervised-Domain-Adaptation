import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from base.base_net import BaseNet
class ResNet50(BaseNet):

    def __init__(self):
        super().__init__()
        self.backbone = ResNet50Fc()
        self.classifier = Classifier()
    def forward(self, x):
        feature = self.backbone(x)
        class_attention, class_logit = self.classifier(feature)
        
        return class_logit, class_attention

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
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        featuremap = self.layer4(x)
        return featuremap
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Conv2d(2048, 31, kernel_size=(1, 1), stride=1)
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        attention = self.layer1(x)
        x = self.gap(attention).reshape(-1,31)
        return attention, x