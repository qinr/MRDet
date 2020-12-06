import torch.nn as nn
from mmcv.cnn.weight_init import normal_init, xavier_init
from torch.nn.modules.utils import _pair
from ..backbones.resnet import Bottleneck
from ..registry import HEADS
from .double_bbox_head import BasicResBlock
from .bbox_head_obb import BBoxHeadOBB


@HEADS.register_module
class DoubleConvFCBBoxHeadOBB(BBoxHeadOBB):


    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(DoubleConvFCBBoxHeadOBB, self).__init__(**kwargs)
        assert self.with_avg_pool
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg


        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels,
                                       self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch()
        # add fc heads
        self.fc_branch = self._add_fc_branch()

        out_dim_reg = 5 if self.reg_class_agnostic else 5 * self.num_classes
        self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)

        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU(inplace=True)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers"""
        branch_convs = nn.ModuleList()
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):
        # conv head
        # The first block increases the number of channels from 256 to 1024
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        cls_score = self.fc_cls(x_fc)

        return cls_score, bbox_pred




