import torch.nn as nn
import torch
from ..utils import convolution
from mmdet.ops.cpools import RightPool, LeftPool, TopPool, BottomPool

class ExtremePooling(nn.Module):

    def __init__(self,
                 in_channels=256,
                 mid_channels=128,
                 out_channels=256):
        super(ExtremePooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.left_conv = convolution(in_channels, mid_channels, 3, padding=1)
        self.right_conv = convolution(in_channels, mid_channels, 3, padding=1)
        self.top_conv = convolution(in_channels, mid_channels, 3, padding=1)
        self.bottom_conv = convolution(in_channels, mid_channels, 3, padding=1)

        self.left_pool = LeftPool()
        self.right_pool = RightPool()
        self.top_pool = TopPool()
        self.bottom_pool = BottomPool()

        self.conv1 = convolution(in_channels, in_channels, 1, with_relu=False)
        self.conv2 = convolution(mid_channels, out_channels, 3, padding=1, with_relu=False)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, x):
        x_left = self.left_conv(x)
        x_left = self.left_pool(x_left)

        x_right = self.right_conv(x)
        x_right = self.right_pool(x_right)

        x_top = self.top_conv(x)
        x_top = self.top_pool(x_top)

        x_bottom = self.bottom_conv(x)
        x_bottom = self.bottom_pool(x_bottom)

        x1 = self.conv2(x_left + x_right + x_bottom + x_top)
        x2 = self.conv1(x)
        x = x1 + x2
        x = self.relu(x)

        return x



