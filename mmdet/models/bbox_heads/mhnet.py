import torch.nn as nn
from mmcv.cnn.weight_init import normal_init, xavier_init
from ..utils import convolution
from ..backbones.resnet import Bottleneck
from ..registry import HEADS
from .double_bbox_head import BasicResBlock
from .bbox_head_obb import BBoxHeadOBB
import torch
from ..plugins import CenterPooling
from ..losses import accuracy
from ..builder import build_loss
import torch.nn.functional as F
from mmdet.core import (auto_fp16,  force_fp32, delta2rec,
                        multiclass_poly_nms_8_points,
                        rbboxRec2Poly)

@HEADS.register_module
class MHNet(BBoxHeadOBB):


    def __init__(self,
                 num_convs_xy=0,
                 num_convs_wh=0,
                 num_fcs_theta=0,
                 num_fcs_cls=0,
                 xy_conv_out_channels=1024,
                 wh_conv_out_channels=1024,
                 theta_fc_out_channels=1024,
                 cls_fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 loss_bbox_xy=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     loss_weight=1.0),
                 loss_bbox_wh=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     loss_weight=1.0),
                 loss_bbox_theta=dict(
                     type='SmoothL1Loss',
                     beta=1.0,
                     loss_weight=1.0),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(MHNet, self).__init__(**kwargs)
        assert self.with_avg_pool
        assert num_convs_xy >= 0
        assert num_convs_wh > 0
        assert num_fcs_theta > 0
        assert num_fcs_cls > 0
        self.num_convs_xy = num_convs_xy
        self.num_convs_wh = num_convs_wh
        self.num_fcs_theta = num_fcs_theta
        self.num_fcs_cls = num_fcs_cls
        self.xy_conv_out_channels = xy_conv_out_channels
        self.wh_conv_out_channels = wh_conv_out_channels
        self.theta_fc_out_channels = theta_fc_out_channels
        self.cls_fc_out_channels = cls_fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.loss_bbox_xy = build_loss(loss_bbox_xy)
        self.loss_bbox_wh = build_loss(loss_bbox_wh)
        self.loss_bbox_theta = build_loss(loss_bbox_theta)

        # xy branch
        self.xy_branch1 = nn.ModuleList()
        for i in range(self.num_convs_xy):
            if i == 0:
                self.xy_branch1.append(
                    convolution(self.in_channels, 256, 3, padding=1))
            else:
                self.xy_branch1.append(
                    convolution(256, 256, 3, padding=1))
        self.xy_center_pool = CenterPooling(256, 128, 256)
        self.xy_branch2 = nn.Sequential(
            convolution(256, 256, 3, padding=1),
            convolution(256, 256, 3, padding=1),
            convolution(256, self.num_classes, 1, with_bn=False, with_relu=False)
        )
        out_dim_xy = 2 if self.reg_class_agnostic else 2 * self.num_classes
        self.fc_xy = nn.Linear(self.num_classes * self.roi_feat_area, out_dim_xy)
        # self.fc_xy = nn.Linear(self.in_channels, out_dim_xy)

        # wh branch
        self.wh_res_block = BasicResBlock(self.in_channels,
                                          self.wh_conv_out_channels)
        self.wh_branch = nn.ModuleList()
        for i in range(self.num_convs_wh):
            self.wh_branch.append(
                convolution(self.wh_conv_out_channels, self.wh_conv_out_channels, 3, padding=1))
        out_dim_wh = 2 if self.reg_class_agnostic else 2 * self.num_classes
        self.fc_wh = nn.Linear(self.wh_conv_out_channels, out_dim_wh)

        # theta branch
        self.theta_branch = self._add_fc_branch(self.num_fcs_theta, self.theta_fc_out_channels)
        out_dim_theta = 1 if self.reg_class_agnostic else self.num_classes
        self.fc_theta = nn.Linear(self.theta_fc_out_channels, out_dim_theta)

        # cls branch
        self.cls_branch = self._add_fc_branch(self.num_fcs_cls, self.cls_fc_out_channels)
        self.fc_cls = nn.Linear(self.cls_fc_out_channels, self.num_classes)
        self.relu = nn.ReLU(inplace=True)


    def _add_conv_branch(self, num_convs, conv_out_channels):
        """Add the fc branch which consists of a sequential of conv layers"""
        branch_convs = nn.ModuleList()
        for i in range(num_convs):
            branch_convs.append(
                Bottleneck(
                    inplanes=conv_out_channels,
                    planes=conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        return branch_convs

    def _add_fc_branch(self, num_fcs, fc_out_channels):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, fc_out_channels))
        return branch_fcs

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_xy, std=0.001)
        normal_init(self.fc_wh, std=0.001)
        normal_init(self.fc_theta, std=0.001)

        for m in self.cls_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

        for m in self.theta_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')

    def forward(self, x_cls, x_reg):
        # cls_branch
        x_cls = x_cls.view(x_cls.size(0), -1)
        for fc in self.cls_branch:
            x_cls = self.relu(fc(x_cls))
        cls_score = self.fc_cls(x_cls)

        # xy_branch
        x_xy = x_reg
        for conv in self.xy_branch1:
            x_xy = conv(x_xy)
        x_xy = self.xy_center_pool(x_xy)
        x_xy = self.xy_branch2(x_xy)
        # x_xy = self.avg_pool(x_xy)
        x_xy = x_xy.view(x_xy.size(0), -1)
        xy_pred = self.fc_xy(x_xy)

        # wh_branch
        x_wh = self.wh_res_block(x_reg)
        for conv in self.wh_branch:
            x_wh = conv(x_wh)
        x_wh = self.avg_pool(x_wh)
        x_wh = x_wh.view(x_wh.size(0), -1)
        wh_pred = self.fc_wh(x_wh)

        # theta_branch
        x_theta = x_reg.view(x_reg.size(0), -1)
        for fc in self.theta_branch:
            x_theta = self.relu(fc(x_theta))
        theta_pred = self.fc_theta(x_theta)

        return cls_score, xy_pred, wh_pred, theta_pred

    def loss(self,
             cls_score,
             bbox_xy_pred,
             bbox_wh_pred,
             bbox_theta_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)

        if bbox_xy_pred is not None and bbox_wh_pred is not None\
                and bbox_theta_pred is not None:
            bbox_xy_targets = bbox_targets[:, :2]
            bbox_wh_targets = bbox_targets[:, 2:4]
            bbox_theta_targets = bbox_targets[:, -1:]
            bbox_xy_weights = bbox_weights[:, :2]
            bbox_wh_weights = bbox_weights[:, 2:4]
            bbox_theta_weights = bbox_weights[:, -1:]
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_xy_pred = bbox_xy_pred.view(bbox_xy_pred.size(0), 2)[pos_inds]
                pos_bbox_wh_pred = bbox_wh_pred.view(bbox_wh_pred.size(0), 2)[pos_inds]
                pos_bbox_theta_pred = bbox_theta_pred.view(bbox_theta_pred.size(0), 1)[pos_inds]
            else:
                pos_bbox_xy_pred = bbox_xy_pred.view(bbox_xy_pred.size(0), -1,
                                               2)[pos_inds, labels[pos_inds]]
                pos_bbox_wh_pred = bbox_wh_pred.view(bbox_wh_pred.size(0), -1,
                                                     2)[pos_inds, labels[pos_inds]]
                pos_bbox_theta_pred = bbox_theta_pred.view(bbox_theta_pred.size(0), -1,
                                                    1)[pos_inds, labels[pos_inds]]
            losses['loss_bbox_xy'] = self.loss_bbox_xy(
                pos_bbox_xy_pred,
                bbox_xy_targets[pos_inds],
                bbox_xy_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
            losses['loss_bbox_wh'] = self.loss_bbox_xy(
                pos_bbox_wh_pred,
                bbox_wh_targets[pos_inds],
                bbox_wh_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
            losses['loss_bbox_theta'] = self.loss_bbox_theta(
                pos_bbox_theta_pred,
                bbox_theta_targets[pos_inds],
                bbox_theta_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses


    @force_fp32(apply_to=('cls_score', 'bbox_xy_pred', 'bbox_wh_pred', 'bbox_theta_pred'))
    def get_det_rbbox2rbbox(self,
                       rrois,
                       cls_score,
                       bbox_xy_pred,
                       bbox_wh_pred,
                       bbox_theta_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        bbox_xy_pred_temp = bbox_xy_pred.view(bbox_xy_pred.size(0), -1, 2)
        bbox_wh_pred_temp = bbox_wh_pred.view(bbox_wh_pred.size(0), -1, 2)
        bbox_theta_pred_temp = bbox_theta_pred.view(bbox_theta_pred.size(0), -1, 1)
        bbox_pred_temp = torch.cat([bbox_xy_pred_temp, bbox_wh_pred_temp, bbox_theta_pred_temp], dim=-1)
        bbox_pred = bbox_pred_temp.view(bbox_pred_temp.size(0), -1)

        if bbox_pred is not None:
            rbboxes_rec = delta2rec(bbox_pred, rrois[:, 1:], self.target_means,
                                           self.target_stds, img_shape)
        else:
            rbboxes_rec = rrois[:, 1:]

        if rescale:
            rbboxes_rec[:, 0::5] /= scale_factor
            rbboxes_rec[:, 1::5] /= scale_factor
            rbboxes_rec[:, 2::5] /= scale_factor
            rbboxes_rec[:, 3::5] /= scale_factor
        rbboxes_poly = rbboxRec2Poly(rbboxes_rec, img_shape)
        if cfg is None:
            return rbboxes_poly, scores
        else:
            det_rbboxes, det_labels = multiclass_poly_nms_8_points(rbboxes_poly, scores,
                                                     cfg.score_thr,
                                                     cfg.nms,
                                                     max_num=cfg.max_per_img)

            return det_rbboxes, det_labels



