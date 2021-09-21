import torch
from torch import nn
from torchvision.models import vgg16


class TextBoxes(nn.Module):
    def __init__(self):
        super(TextBoxes, self).__init__()
        vgg = vgg16(pretrained=True)
        self.backbone = vgg
    
    def forward(self):
        pass
