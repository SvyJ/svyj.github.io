# coding=utf-8
import torch
import torch.nn as nn
from torchsummary import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256*6*6, 4096),
        #     nn.ReLU(inplace=True),

        #     nn.Dropout(),
        #     nn.Linear(4096, 1024),
        #     nn.ReLU(inplace=True),

        #     nn.Linear(1024, 2)
        # )

    def forward(self, x): 
        # print(x.size())
        x = self.features(x) 
        # print(x.size())
        # x = x.view(x.size(0), 256*6*6)
        # x = self.classifier(x)
        return x

net = AlexNet().to(device)
summary(net, input_size=(3, 500, 500))