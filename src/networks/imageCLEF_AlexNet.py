import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from base.base_net import BaseNet

class ImageCLEFAlexNet(BaseNet):

    def __init__(self):
        super().__init__()
        self.backbone = AlexNet()
        self.classifier1 = nn.Linear(4096, 31)
    def forward(self, x):
        feature = self.backbone(x)
        x = self.classifier1(feature)
        return  feature,x
    
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=True)
        self.features  = model.features 
        self.avgpool = model.avgpool 
        self.classifier = model.classifier[:-1]
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier1 = nn.Linear(4096, 1024)
        self.classifier2 = nn.Linear(1024, 31)
    def forward(self, x):
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x