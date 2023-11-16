import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down=True):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channel, prev_channel, out_channel, up=True):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + prev_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x, res=None):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DUNet(nn.Module):
    def __init__(self):
        super(DUNet, self).__init__()

        # 64 128 256 256
        self.down1 = DownBlock(3, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 256)
        self.down5 = DownBlock(256, 256)

        self.up1 = UpBlock(256, 256, 256)
        self.up2 = UpBlock(256, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)

        self.upsample1 = nn.Upsample(size=(37, 37), mode='bilinear')
        self.upsample2 = nn.Upsample(size=(74, 74), mode='bilinear')
        self.upsample3 = nn.Upsample(size=(149, 149), mode='bilinear')
        self.upsample4 = nn.Upsample(size=(299, 299), mode='bilinear')

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, input):
        maps1 = self.down1(input)
        x = self.maxpool(maps1)

        maps2 = self.down2(x)
        x = self.maxpool(maps2)

        maps3 = self.down3(x)
        x = self.maxpool(maps3)

        maps4 = self.down4(x)
        x = self.maxpool(maps4)

        x = self.down5(x)

        x = self.upsample1(x)
        x = torch.cat((x, maps4), dim=1)
        x = self.up1(x)

        x = self.upsample2(x)
        x = torch.cat((x, maps3), dim=1)
        x = self.up2(x)

        x = self.upsample3(x)
        x = torch.cat((x, maps2), dim=1)
        x = self.up3(x)

        x = self.upsample4(x)
        x = torch.cat((x, maps1), dim=1)
        x = self.up4(x)

        return self.conv(x)
