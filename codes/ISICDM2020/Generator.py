import torch
import torch.nn as nn
from .UNet import UNet
from .ZZNet import ZZNet
from .CSNet import CSNet
from .Attention_UNet import Attention_UNet
from .DA_Res50 import DA_ResNet50
from .UNet_Nested import UNet_Nested
from .SBUNet import SBUNet
from .XZNet import XZNet
from .UNet_Inception import UNet_Inception
from .XZNet_Inception import XZNet_Inception


class Generator(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(Generator, self).__init__()
        # self.G = UNet(in_channels, num_classes)
        # self.G = CSNet(in_channels, num_classes)
        # self.G = Attention_UNet(in_channels, num_classes)
        # self.G = SBUNet(in_channels, num_classes)
        # self.G = XZNet(in_channels, num_classes)
        # self.G = DA_ResNet50()
        # self.G = UNet_Nested(in_channels=1, num_classes=2)
        self.G = UNet_Inception(in_channels, num_classes)
        # self.G = XZNet_Inception(in_channels, num_classes)
        # self.G = ZZNet(in_channels, (512, 512), num_classes)

    def forward(self, x):
        x = self.G(x)
        return x