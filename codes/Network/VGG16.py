# coding=utf-8
import torch
import torch.nn as nn
from torchsummary import summary

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                # nn.ReLU(inplace = True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace = True)
            )

    def forward(self, x):
        return self.conv(x)


class TripleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv = nn.Sequential(
                # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                # nn.ReLU(inplace = True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace = True),
                nn.Conv2d(out_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace = True)
            )

    def forward(self, x):
        return self.conv(x)


class VGG16(nn.Module):

    def __init__(self, in_channels):
        super(VGG16, self).__init__()
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = TripleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = TripleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = TripleConv(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512*3*4, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, 7)
        )
    
    def forward(self, x): 
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool4(x)
        
        # print(x.shape)
        x = x.view(x.size(0), 512*3*4)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG16(in_channels=1).to(device)
print(model)
summary(model, input_size=(1, 112, 144))
