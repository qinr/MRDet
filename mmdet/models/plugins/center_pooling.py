# written by qr
import torch.nn as nn
import torch
from ..utils import convolution

class CenterPooling(nn.Module):

    def __init__(self,
                 in_channels=256,
                 mid_channels=128,
                 out_channels=256):
        super(CenterPooling,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels

        self.conv1 = convolution(in_channels, mid_channels, 3, padding=1)
        self.conv2 = convolution(in_channels, mid_channels, 3, padding=1)
        self.conv3 = convolution(in_channels, out_channels, 1, with_relu=False)

        self.conv4 = convolution(mid_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

    def init_weights(self):
        pass

    def forward(self, feature):
        x1 = self.conv1(feature)
        x1_pool = torch.max(x1, dim=-1)[0]
        x1_pool = x1_pool.unsqueeze(3).expand_as(x1)

        x2 = self.conv2(feature)
        x2_pool = torch.max(x2, dim=2)[0]
        x2_pool = x2_pool.unsqueeze(2).expand_as(x2)

        x_pool = x1_pool + x2_pool
        x = self.conv4(x_pool)

        x3 = self.conv3(feature)
        x = x + x3
        x = self.relu(x)
        return x



