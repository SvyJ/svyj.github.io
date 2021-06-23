import torch 
import torch.nn as nn
from torchsummary import summary
from VGG19 import *


# without bn version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP,self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1)) #(1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class DoubleUNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(DoubleUNet, self).__init__()
        self.VGG19_ = VGG19(make_layers(cfg['E']))

    def forward(self, x):
        output = self.VGG19_(x)
        x5 = output['x5']  
        x4 = output['x4']  
        x3 = output['x3']  
        x2 = output['x2']  
        x1 = output['x1'] 
        print(x1.shape, x2.shape, x3.shape, x4.shape, x5.shape)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DoubleUNet(in_channels=1, num_classes=4).to(device)
print(model)
summary(model, input_size=(3, 512, 512)) 