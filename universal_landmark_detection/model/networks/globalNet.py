import torch
import torch.nn as nn
import torch.nn.functional as F


class myConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(myConv2d, self).__init__()
        padding = (kernel_size-1)//2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.conv(x)


class dilatedConv(nn.Module):
    ''' stride == 1 '''

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(dilatedConv, self).__init__()
        # f = (kernel_size-1) * d +1
        # new_width = (width - f + 2 * padding)/stride + stride
        padding = (kernel_size-1) * dilation // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GlobalNet(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=0.25, kernel_size=3, dilations=None):
        super(GlobalNet, self).__init__()
        self.scale_factor = scale_factor
        if not isinstance(in_channels, list):
            in_channels = [in_channels]
        if not isinstance(out_channels, list):
            out_channels = [out_channels]
        mid_channels = 128
        if dilations is None:
            dilations = [1, 2, 5]
        for i, n_chan in enumerate(in_channels):
            setattr(self, 'in{i}'.format(i=i),
                    myConv2d(n_chan, mid_channels, 3))
        for i, n_chan in enumerate(out_channels):
            setattr(self, 'out{i}'.format(i=i),
                    myConv2d(mid_channels, n_chan, 1))
            convs = [dilatedConv(mid_channels, mid_channels,
                                 kernel_size, dilation) for dilation in dilations]
            convs = nn.Sequential(*convs)
            setattr(self, 'convs{}'.format(i), convs)

    def forward(self, x, task_idx=0):
        size = x.size()[2:]
        sf = self.scale_factor
        x = F.interpolate(x, scale_factor=sf)
        x = getattr(self, 'in{}'.format(task_idx))(x)
        x = getattr(self, 'convs{}'.format(task_idx))(x)
        x = getattr(self, 'out{}'.format(task_idx))(x)
        x = F.interpolate(x, size=size)
        return {'output': torch.sigmoid(x)}
