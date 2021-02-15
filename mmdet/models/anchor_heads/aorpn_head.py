from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, orient_anchor_target, force_fp32,
                        multi_apply, target2poly, delta2bbox, hbbox2rec)
from ..builder import build_loss
from ..registry import HEADS
import torch.nn.functional as F
from mmdet.ops import poly_nms

@HEADS.register_module
class AO_RPNHead(nn.Module):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).

    Args:
        num_classes (int): Number of categories including the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_scales (Iterable): Anchor scales.
        anchor_ratios (Iterable): Anchor aspect ratios.
        anchor_strides (Iterable): Anchor strides.
        anchor_base_sizes (Iterable): Anchor base sizes.
        target_means (Iterable): Mean values of regression targets.
        target_stds (Iterable): Std values of regression targets.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 anchor_scales=[8, 16, 32],
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means_hbb=(.0, .0, .0, .0),
                 target_stds_hbb=(1.0, 1.0, 1.0, 1.0),
                 target_means_obb=(.0, .0, .0, .0),
                 target_stds_obb=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_obb=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(AO_RPNHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels # 在子类中使用，例如RPNHead
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means_hbb = target_means_hbb
        self.target_stds_hbb = target_stds_hbb
        self.target_means_obb = target_means_obb
        self.target_stds_obb = target_stds_obb

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obb = build_loss(loss_obb)
        self.fp16_enabled = False

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))
            # anchor生成器，可生成给定scale/ratios的anchor，并将其变换到单位为anchor_base的格子上

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.conv_cls = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.cls_out_channels, 1)
        self.conv_hbb = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.conv_obb = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.conv, std=0.01)
        normal_init(self.conv_cls, std=0.01) # rand_normal_init
        normal_init(self.conv_hbb, std=0.01)
        normal_init(self.conv_obb, std=0.01)

    def forward_single(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_hbb(x)
        obb_pred = self.conv_obb(x)
        return cls_score, bbox_pred, obb_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        # anchor_list:每张特征图的anchor(len(scales) * len(ratios) * feat_w * feat_h个）

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list
        # 返回每张图的anchor，以及anchor是否有效的flag，是否有效是根据bbox是否超出图片来计算的

    def loss_single(self, cls_score, bbox_pred, obb_pred, labels,
                    label_weights, bbox_targets, bbox_weights,
                    obb_targets, obb_weights, num_total_samples, cfg):
        # label_weights/bbox_weights:loss内的权重
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        obb_targets = obb_targets.reshape(-1, 4)
        obb_weights = obb_weights.reshape(-1, 4)
        obb_pred = obb_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_obb = self.loss_obb(
            obb_pred,
            obb_targets,
            obb_weights,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox, loss_obb

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'obb_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             obb_preds,
             gt_bboxes,
             gt_rbboxes_poly,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        # 生成anchor，将anchor与gt_bbox匹配，生成正负样本，计算gt_delta(anchor与其对应的gt_bbox的delta)
        # 得到每个anchor的label/label_weights/delta/delta_weights，以及pos_inds/neg_inds
        # 利用这些使用交叉熵和SmoothL1Loss，得到loss
        # 其中，get_anchor完成了anchor生成
        # anchor_target完成了anchor与gt_bbox匹配(assigner) + 正负样本生成（sampler）+ gt_delta计算
        # + 得到每个anchor的label/label_weights/delta/delta_weights，以及pos_inds/neg_inds（返回值）
        # loss_single根据anchor_target的返回值完成了loss计算
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores] # 特征图的w/h
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        # anchor_list:每张图的anchor
        # valid_flag_list:anchor是否有效的flag，是否有效是根据bbox是否超出图片来计算的
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # anchor_target: 将anchor和gt_bbox匹配，得到正样本和负样本,并用sampler将这些结果进行封装，方便之后使用
        # torch.cuda.empty_cache()
        cls_reg_targets = orient_anchor_target(
            bbox_preds,
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            gt_rbboxes_poly,
            img_metas,
            self.target_means_hbb,
            self.target_stds_hbb,
            self.target_means_obb,
            self.target_stds_obb,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        # cls_reg_target包含：
        # labels:每个anchor对应的label
        # label_weights:每个anchor cls_loss的权重，负样本权重为1，正样本权重可为1也可为其他值
        # bbox_targets：每个anchor与其对应的gt_bbox之前的delta，用于回归
        # bbox_weights: 每个anchor bbox_reg的权重，正样本为1，负样本为0
        # pos_inds：anchor中正样本的索引
        # neg_inds: anchor中负样本的索引
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, obb_targets_list, obb_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox, losses_obb = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            obb_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            obb_targets_list,
            obb_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        del cls_reg_targets, labels_list, label_weights_list, \
            bbox_targets_list, bbox_weights_list, obb_targets_list, obb_weights_list
        del anchor_list, valid_flag_list
        return dict(loss_orient_anchor_cls=losses_cls, loss_orient_anchor_bbox=losses_bbox, loss_orient_anchor_obb=losses_obb)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'obb_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   obb_preds,
                   img_metas,
                   cfg,
                   rescale=False):
        # 根据cls_scores + bbox_preds + anchors（所有，不筛选，无正负样本的区分）得到proposals，并完成NMS
        # 其中，get_bboxes_single中完成了proposals生成+NMS
        """
        Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): size / scale info for each image
            cfg (mmcv.Config): test / postprocessing configuration
            rescale (bool): if True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(num_classes=9, in_channels=1)
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores) # 层数：如果有fpn,可能传进来多层特征图

        device = cls_scores[0].device
        mlvl_anchors = [
            self.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.anchor_strides[i],
                device=device) for i in range(num_levels)
        ]
        # m1v1_anchors：生成所有的anchors，对应不同层的特征图+locations+scales+ratios
        # 这里“不同层的特征图”主要是在尺寸，并没有用到特征图具体的值，只是根据不同特征图的尺寸生成了不同anchor
        # anchor的生成不会用的特征图的值，只会用到特征图的尺寸
        result_list = []
        # 生成每张图的proposals
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            obb_pred_list = [
                obb_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            # cls_score_list:这张图片所有特征图的cls_score
            # bbox_pred_list:这张图片所有特征图的bbox_pred
            # m1v1_anchor:所有特征图的所有anchors，仅与特征图尺寸有关，与具体的输入图片无关，因此每张输入图片使用的m1v1_anchor是一样的
            # todo:r_nms，bbox2delta以及delta2bbox等需要修改，适应带方向
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               obb_pred_list,
                                               mlvl_anchors, img_shape,
                                               scale_factor, cfg, rescale)
            # get_bboxes_single:由anchor和bbox_pred得到proposals，并实现NMS
            result_list.append(proposals)
        return result_list
        # 得到经过NMS的proposals

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          obb_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        mlvl_proposals_rotate = []
        # m1v1_proposals = []
        for idx in range(len(cls_score_list)):
            cls_score = cls_score_list[idx]
            bbox_pred = bbox_pred_list[idx]
            obb_pred = obb_pred_list[idx]
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            cls_score = cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                cls_score = cls_score.reshape(-1)
                scores = cls_score.sigmoid()
            else:
                cls_score = cls_score.reshape(-1, 2)
                scores = cls_score.softmax(dim=1)[:, 1]
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            obb_pred = obb_pred.permute(1, 2, 0).reshape(-1, 4)
            # nms_pre:NMS前，选出置信度前nms_pre高的anchor
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)  # 置信度前nms_pre高的anchor inds
                bbox_pred = bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
                obb_pred = obb_pred[topk_inds, :]
            # 将选出的anchor由delta变换得到proposals(x1,y1,x2,y2)
            proposals = delta2bbox(anchors, bbox_pred, self.target_means_hbb,
                                     self.target_stds_hbb)
            proposals_rec = hbbox2rec(proposals)
            if cfg.min_bbox_size > 0:
                w = proposals_rec[:, 2]
                h = proposals_rec[:, 3]
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                # proposals = proposals[valid_inds, :]
                proposals_rec = proposals_rec[valid_inds, :]
                scores = scores[valid_inds]
            proposals_rotate = target2poly(proposals_rec, obb_pred, img_shape,
                                       self.target_means_obb, self.target_stds_obb)
            proposals_rotate = torch.cat([proposals_rotate, scores.unsqueeze(-1)], dim=-1)
            proposals_rotate, _ = poly_nms(proposals_rotate, cfg.nms_thr)  # 根据nms_thr完成NMS
            # proposals = proposals[_, :]
            proposals_rotate = proposals_rotate[:cfg.nms_post, :]  # 选出置信度前nms_post的proposals
            # proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals_rotate.append(proposals_rotate)
            # m1v1_proposals.append(proposals)
        proposals_rotate = torch.cat(mlvl_proposals_rotate, 0)
        # proposals = torch.cat(m1v1_proposals, 0)
        if cfg.nms_across_levels:
            proposals_rotate, _ = poly_nms(proposals_rotate, cfg.nms_thr)
            proposals = proposals[_, :]
            proposals_rotate = proposals_rotate[:cfg.max_num, :]
            # proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals_rotate[:, 8]
            num = min(cfg.max_num, proposals_rotate.shape[0])
            _, topk_inds = scores.topk(num)
            proposals_rotate = proposals_rotate[topk_inds, :]
            # proposals = proposals[topk_inds, :]
        return proposals_rotate #, proposals