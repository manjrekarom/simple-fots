import math
from model.resnetfpn import ResnetFPN
import torch
from torch import nn


class TextDet(nn.Module):
    # reference: https://github.com/SakuraRiven/EAST/blob/cec7ae98f9c21a475b935f74f4c3969f3a989bd4/model.py#L136
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        # fg / bg 
        self.conv2 = nn.Conv2d(256, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        # bounding box
        self.conv3 = nn.Conv2d(256, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        # TODO: find the range of bounding box co-ordinates
        # self.scope = ___
        # orientation
        self.conv4 = nn.Conv2d(256, 1, 1)
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        score = self.sigmoid1(self.conv2(x))
        # TODO: Convert from 0-1 to 0-"w or h"
        # loc of top, bot, left, right sides of the bounding bo
        loc = self.sigmoid2(self.conv3(x))
        angle = (self.sigmoid3(self.conv4(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), axis=1)
        return score, geo


class RoiRotate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t, b, l, r, ht=8):
        s = ht / (t + b)


class FOTS(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=False):
        super().__init__()
        self.fpn = ResnetFPN(arch=backbone, pretrained=pretrained)
        self.fpn.create_architecture()  # this is stupidity; remove this later
        self.text_det = TextDet()

    def forward(self, x):
        shared_features = self.fpn(x)[0]
        text_det = self.text_det(shared_features)
        return text_det
