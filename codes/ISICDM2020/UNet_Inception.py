import torch
import torch.nn as nn
import torchsummary

 
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True)  
            )

    def forward(self, x):
        return self.conv(x)


class InceptionModule(nn.Module):

    def __init__(self, in_channels, out_channels_1, reduce_3, out_channels_3, reduce_5, out_channels_5, pool_proj):
        super(InceptionModule, self).__init__()
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels_1),
            nn.ReLU(inplace=True)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(reduce_3),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3, out_channels_3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels_3),
            nn.ReLU(inplace=True)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(reduce_5),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5, out_channels_5, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels_5),
            nn.ReLU(inplace=True)
        )
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Conv2d(out_channels_1+out_channels_3+out_channels_5+pool_proj, in_channels, kernel_size=1, bias=False)
    def forward(self, x):
        # print(self.branch_1(x).shape, self.branch_2(x).shape, self.branch_3(x).shape, self.branch_4(x).shape)
        return self.conv(torch.cat([self.branch_1(x), self.branch_2(x), self.branch_3(x), self.branch_4(x)], dim=1))


class UNet_Inception(nn.Module):

    def __init__(self, in_channels, num_classes): # <<----------------------------------------改这里
        super(UNet_Inception,self).__init__()
        # Conv
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) 
                                                    #       #        #  #
        self.inception_block1 = InceptionModule(64, 32, 16, 16, 16, 16, 32)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.inception_block2 = InceptionModule(128, 64, 32, 32, 32, 32, 64)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.inception_block3 = InceptionModule(256, 128, 64, 64, 64, 64, 128)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.inception_block4 = InceptionModule(512, 256, 128, 128, 128, 128, 256)

        self.conv5 = DoubleConv(512, 1024)

        

        # DeConv
        self.up_conv6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up_conv7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up_conv8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up_conv9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        
        self.conv10 = nn.Conv2d(64, num_classes, 1)
        
    
    def forward(self,x):
        conv_out_1 = self.conv1(x)
        pool_out_1 = self.pool1(conv_out_1)
        conv_out_2 = self.conv2(pool_out_1)
        pool_out_2 = self.pool2(conv_out_2)
        conv_out_3 = self.conv3(pool_out_2)
        pool_out_3 = self.pool3(conv_out_3)
        conv_out_4 = self.conv4(pool_out_3)
        pool_out_4 = self.pool4(conv_out_4)
        conv_out_5 = self.conv5(pool_out_4)
        up_conv_out_6 = self.up_conv6(conv_out_5)
        concate_6 = torch.cat([up_conv_out_6, self.inception_block4(conv_out_4)], dim=1) 

        conv_out_6 = self.conv6(concate_6)
        up_conv_out_7 = self.up_conv7(conv_out_6)
        concate_7 = torch.cat([up_conv_out_7, self.inception_block3(conv_out_3)], dim=1)

        conv_out_7 = self.conv7(concate_7)
        up_conv_out_8 = self.up_conv8(conv_out_7)
        concate_8 = torch.cat([up_conv_out_8, self.inception_block2(conv_out_2)], dim=1)

        conv_out_8 = self.conv8(concate_8)
        up_conv_out_9 = self.up_conv9(conv_out_8)
        concate_9 = torch.cat([up_conv_out_9, self.inception_block1(conv_out_1)], dim=1)

        conv_out_9 = self.conv9(concate_9)
        conv_out_10 = self.conv10(conv_out_9)
        # out = nn.Sigmoid()(conv_out_10) 
        return conv_out_10


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet_Inception(in_channels=1, num_classes=2).to(device)
# # model = ZZNet(in_channels=1, input_size=(150, 194), num_classes=4).to(device)
# # print(model)
# summary(model, input_size=(1, 512, 512)) 

# dummy_input = torch.rand(8, 1, 512, 512).to(device)


