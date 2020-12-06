# written by qr
from ..utils import build_conv_layer
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import multi_apply

class LargeSeperateConv(nn.Module):

    def __init__(self,
                 in_channels=2048,
                 mid_channels=256,
                 out_channels=10*7*7,
                 k=15,
                 conv_cfg=None,
                 mode=0
                ):
        # mode表示两种模式，这两种模式中relu的位置不同
        # mode=0: light head rcnn中使用的模式
        # mode=1: ROI transformer中使用的模式
        super(LargeSeperateConv,self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.mode = mode

        self.conv11 = build_conv_layer(
            conv_cfg,
            in_channels,
            mid_channels,
            kernel_size=(k, 1),
            stride=1,
            padding=(k//2, 0)
        )
        self.conv12 = build_conv_layer(
            conv_cfg,
            mid_channels,
            out_channels,
            kernel_size=(1, k),
            stride=1,
            padding=(0, k//2)
        )
        self.conv21 = build_conv_layer(
            conv_cfg,
            in_channels,
            mid_channels,
            kernel_size=(1, k),
            stride=1,
            padding=(0, k//2)
        )
        self.conv22 = build_conv_layer(
            conv_cfg,
            mid_channels,
            out_channels,
            kernel_size=(k, 1),
            stride=1,
            padding=(k//2, 0)
        )
        self.relu = nn.ReLU(inplace=True)

    def init_weights(self):
        normal_init(self.conv11, std=0.01)
        normal_init(self.conv12, std=0.01)
        normal_init(self.conv21, std=0.01)
        normal_init(self.conv22, std=0.01)

    def forward(self, x):

        if self.mode == 0:
            out1 = self.conv11(x)
            out1 = self.conv12(out1)

            out2 = self.conv21(x)
            out2 = self.conv22(out2)

            out = out1 + out2
            out = self.relu(out)
        else:
            out1 = self.conv11(x)
            out1 = self.relu(out1)
            out1 = self.conv12(out1)
            out1 = self.relu(out1)

            out2 = self.conv21(x)
            out2 = self.relu(out2)
            out2 = self.conv22(out2)
            out2 = self.relu(out2)

            out = out1 + out2

        return out


