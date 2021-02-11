import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from base.base_net import BaseNet

class Office31ResNet(BaseNet):

    def __init__(self):
        super().__init__()
#         self.s_centroid = torch.zeros(31, 2048)
#         self.t_centroid = torch.zeros(31, 2048)
        self.backbone = ResNet50Fc()
        self.backbone = torch.nn.DataParallel(self.backbone, device_ids = [1,0])
        self.backbone.to("cuda:1")
        self.classifier1 = nn.Linear(2048, 31)
        self.classifier1 = torch.nn.DataParallel(self.classifier1, device_ids = [1,0])
        self.classifier1.to("cuda:1")
#         self.backbone = ResNet50Fc()
#         self.classifier1 = nn.Linear(2048, 31)
#         self.classifier2 = nn.Linear(2048, 31)
#         self.projectionhead = Projection()
#         self.classifier1 = Classifier()
#         self.discriminator = nn.Linear(2048, 1)
#         self.discriminator = AdversarialNetwork(2048)
    def forward(self, x):
        feature = self.backbone(x)
        out1 = self.classifier1(feature)
        return  feature,out1
# class Projection(nn.Module):
#     def __init__(self):
#         super(Projection, self).__init__()
#         self.ad_layer1 = nn.Linear(2048,1024)
# #         self.ad_layer2 = nn.Linear(1024,512)
# #         self.bn2 = nn.BatchNorm1d(1024)
# #         self.ad_layer3 = nn.Linear(1024,1)
#         self.relu1 = nn.ReLU()

#     def forward(self, x):
#         x = self.ad_layer1(x)
# #         x = self.relu1(x)
# #         x = self.ad_layer2(x)
# #         x = self.bn2(x)
# #         x = self.relu2(x)
# #         x = self.ad_layer3(x)
# #         x = self.sigmoid(x)
#         return x

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
        self.gap1 = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap1(x).reshape(-1,2048)
        return x

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(2048,1024)
        self.bn = nn.BatchNorm1d(1024)
        self.ad_layer2 = nn.Linear(1024,1)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.ad_layer3 = nn.Linear(1024,1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = GradReverse.grad_reverse(x, constant)
        x = self.ad_layer1(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.ad_layer2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.ad_layer3(x)
#         x = self.sigmoid(x)
        return x

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ad_layer1 = nn.Linear(2048,1024)
        self.ad_layer2 = nn.Linear(1024,31)
#         self.bn2 = nn.BatchNorm1d(1024)
#         self.ad_layer3 = nn.Linear(1024,1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.ad_layer2(x)
#         x = self.bn2(x)
#         x = self.relu2(x)
#         x = self.ad_layer3(x)
#         x = self.sigmoid(x)
        return x


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