import torch.nn as nn
from mmcv.cnn import constant_init, kaiming_init


class convolution(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 groups=1,
                 with_bn=True,
                 with_relu=True):
        super(convolution, self).__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=not with_bn)
        self.with_bn = with_bn
        if self.with_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.with_relu = with_relu
        if self.with_relu:
            self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        kaiming_init(self.conv, nonlinearity='relu')
        if self.with_bn:
            constant_init(self.bn, 1, bias=0)

    def forward(self, x):
        x = self.conv(x)
        if self.with_bn:
            x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x

