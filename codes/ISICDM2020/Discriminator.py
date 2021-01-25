import torch
import torch.nn as nn


class FCDiscriminator(nn.Module):

    def __init__(self, in_channels, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv0 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1)# 64*72*56
        self.conv1 = nn.Conv2d(in_channels, ndf, kernel_size=3, stride=2, padding=1)  # 64*72*56
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1)      # 128*36*28
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)    # 256*18*14
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)    # 512*9*7
 
        self.classifier = nn.Linear(512*4*4, 1)

        self.avgpool = nn.AvgPool2d((7, 7))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, map, feature):
        '''
        map: mask or generated mask
        feature: original img
        '''
        map_feature = self.conv0(map)
        image_feature = self.conv1(feature)
        x = torch.add(map_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_num_ch, out_num_ch, filter_size=4, stride=2, padding=1, activation='lrelu'):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_num_ch, out_num_ch, filter_size, stride, padding=padding),
            nn.BatchNorm2d(out_num_ch)
            )
        if activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        if activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        if activation == 'elu':
            self.act = nn.ELU(inplace=True)
        else:
            self.act = nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, in_num_ch, first_num_ch=64):
        super(Discriminator, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_num_ch, first_num_ch, 4, 2, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_2 = ConvBlock(first_num_ch, 2*first_num_ch)
        self.conv_3 = ConvBlock(2*first_num_ch, 4*first_num_ch)
        self.conv_4 = ConvBlock(4*first_num_ch, 8*first_num_ch, stride=1)
        self.output = nn.Conv2d(8*first_num_ch, 1, 4)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        conv_3 = self.conv_3(conv_2)
        conv_4 = self.conv_4(conv_3)
        output = self.output(conv_4)
        output_act = self.output_act(output)
        return output_act