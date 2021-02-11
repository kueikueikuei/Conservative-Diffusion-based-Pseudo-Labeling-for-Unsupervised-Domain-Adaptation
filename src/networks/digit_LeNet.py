import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from base.base_net import BaseNet
from .grad_Reverse import GradReverse
class DigitLeNet(BaseNet):

    def __init__(self):
        super().__init__()
        self.backbone = Feature()
        self.classifier1 = nn.Linear(100, 10)
        self.classifier2 = nn.Linear(100, 10)
        self.s_centroid = torch.zeros(10, 100)
        self.t_centroid = torch.zeros(10, 100)
        self.domain_classifier = AdversarialNetwork(in_feature=100)
    def forward(self, x):
        feature = self.backbone(x)
        class_logit1 = self.classifier1(feature)
        class_logit2 = self.classifier2(feature)
        return feature, class_logit1, class_logit2


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*5*5, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Linear(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=2, kernel_size=2, dilation=(1, 1))
        x = x.view(x.size(0), 48*5*5)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.relu(self.bn2_fc(self.fc2(x)))
        return x   
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature,100)
        self.ad_layer2 = nn.Linear(100,1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,constant):
        x = GradReverse.grad_reverse(x, constant)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.ad_layer2(x)
        x = self.sigmoid(x)
        return x

    def output_num(self):
        return 1
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)