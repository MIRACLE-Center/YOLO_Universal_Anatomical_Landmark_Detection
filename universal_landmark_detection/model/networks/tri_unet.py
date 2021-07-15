import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, task_num=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(mid_channels)
                                  for i in range(task_num)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(out_channels)
                                  for i in range(task_num)])
        self.conv1 = nn.ModuleList([nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1) for i in range(task_num)])
        self.conv2 = nn.ModuleList([nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1) for i in range(task_num)])

    def forward(self, x, task_idx=0):
        x = self.conv1[task_idx](x)
        x = self.relu1(self.bn1[task_idx](x))
        x = self.conv2[task_idx](x)
        x = self.relu2(self.bn2[task_idx](x))
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, task_num=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, task_num=task_num)

    def forward(self, x, task_idx=0):
        return self.conv(self.maxpool(x), task_idx)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, task_num=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, task_num)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(
                in_channels, out_channels, task_num=task_num)

    def forward(self, x1, x2, task_idx=0):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, task_idx)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Tri_UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Tri_UNet, self).__init__()
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.task_num = len(in_channels)

        for i, (n_chan, n_class) in enumerate(zip(in_channels, out_channels)):
            setattr(self, 'in{i}'.format(i=i), OutConv(n_chan, 64))
            setattr(self, 'out{i}'.format(i=i), OutConv(64, n_class))
        self.conv = DoubleConv(64, 64, task_num=self.task_num)
        self.down1 = Down(64, 128, task_num=self.task_num)
        self.down2 = Down(128, 256, task_num=self.task_num)
        self.down3 = Down(256, 512, task_num=self.task_num)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, task_num=self.task_num)
        self.up1 = Up(1024, 512 // factor, bilinear, task_num=self.task_num)
        self.up2 = Up(512, 256 // factor, bilinear, task_num=self.task_num)
        self.up3 = Up(256, 128 // factor, bilinear, task_num=self.task_num)
        self.up4 = Up(128, 64, bilinear, task_num=self.task_num)

    def forward(self, x, task_idx=0):
        x1 = getattr(self, 'in{}'.format(task_idx))(x)
        x1 = self.conv(x1, task_idx)
        x2 = self.down1(x1, task_idx)
        x3 = self.down2(x2, task_idx)
        x4 = self.down3(x3, task_idx)
        x5 = self.down4(x4, task_idx)
        x = self.up1(x5, x4, task_idx)
        x = self.up2(x, x3, task_idx)
        x = self.up3(x, x2, task_idx)
        x = self.up4(x, x1, task_idx)
        logits = getattr(self, 'out{}'.format(task_idx))(x)
        return {'output': torch.sigmoid(logits)}
