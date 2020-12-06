import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, force_fp32, delta2rec_v1,
                        rbboxRec2Poly_v1, multiclass_poly_nms_rec,
                        choose_best_rroi, bbox_target_rbbox, rbbox_target_rbbox,
                        delta2rec_v2, hbbox2rbboxRec_v1,
                        hbbox2rbboxRec_v2, multiclass_poly_nms_8_points, rbboxRec2Poly_v2)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS


@HEADS.register_module
class BBoxHeadOBB(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=81,
                 target_means=[0., 0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
                 reg_class_agnostic=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(BBoxHeadOBB, self).__init__()
        assert with_cls or with_reg
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg


        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        in_channels = self.in_channels # light-head-rcnn:10
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area # light-head-rcnn:10*7*7
        if self.with_cls:
            self.fc_cls = nn.Linear(in_channels, num_classes)
            # 会将特征图展开，这里的in_channels = self.inchannels * roi_feat_area
        if self.with_reg:
            out_dim_reg = 5 if reg_class_agnostic else 5 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        self.debug_imgs = None

    def init_weights(self):
        if self.with_cls:
            nn.init.normal_(self.fc_cls.weight, 0, 0.01)
            nn.init.constant_(self.fc_cls.bias, 0)
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    def get_target_hbbox2rbbox(self, sampling_results, gt_rbboxes_poly, cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        # pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]

        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_gt_labels,
            pos_assigned_gt_inds,
            gt_rbboxes_poly,
            cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)

        # print(bbox_targets)
        return cls_reg_targets


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss_hbbox2rbbox(self,
             cls_score,
             bbox_pred,
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
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               5)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    def get_det_hbbox2rbbox(self,
                       hrois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if hrois.size(1) == 5:
            # hbboxes_rec = hbbox2rbboxRec_v2(hrois[:, 1:])
            hbboxes_rec = hbbox2rbboxRec_v2(hrois[:, 1:])
        elif hrois.size(1) == 6:
            hbboxes_rec = hrois[:, 1:]
        else:
            assert hrois.size(1) != 5 and hrois.size(1) != 6, "hrois.size(1) must be 5 or 6"

        if bbox_pred is not None:
            rbboxes_rec = delta2rec_v2(bbox_pred, hbboxes_rec, self.target_means,
                                           self.target_stds, img_shape)
        else:
            rbboxes_rec = hbboxes_rec

        if rescale:
            rbboxes_rec[:, 0::5] /= scale_factor
            rbboxes_rec[:, 1::5] /= scale_factor
            rbboxes_rec[:, 2::5] /= scale_factor
            rbboxes_rec[:, 3::5] /= scale_factor
        rbboxes_poly = rbboxRec2Poly_v2(rbboxes_rec, img_shape)
        if cfg is None:
            return rbboxes_rec, scores
        else:
            det_rbboxes, det_labels = multiclass_poly_nms_8_points(rbboxes_poly, scores,
                                                                   cfg.score_thr,
                                                                   cfg.nms,
                                                                   max_num=cfg.max_per_img)

            return det_rbboxes, det_labels

    def get_target_rbbox2rbbox(self, sampling_results, gt_rbboxes_rec, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = rbbox_target_rbbox(
            pos_proposals,
            neg_proposals,
            pos_assigned_gt_inds,
            gt_rbboxes_rec,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets

    def loss(self,
             cls_score,
             bbox_pred,
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

        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 5)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               5)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)

        return losses


    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_det_rbbox2rbbox(self,
                       rrois,
                       cls_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))
        scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            rbboxes_rec = delta2rec_v2(bbox_pred, rrois[:, 1:], self.target_means,
                                           self.target_stds, img_shape)
        else:
            rbboxes_rec = rrois[:, 1:]

        if rescale:
            rbboxes_rec[:, 0::5] /= scale_factor
            rbboxes_rec[:, 1::5] /= scale_factor
            rbboxes_rec[:, 2::5] /= scale_factor
            rbboxes_rec[:, 3::5] /= scale_factor
        rbboxes_poly = rbboxRec2Poly_v2(rbboxes_rec, img_shape)
        if cfg is None:
            return rbboxes_rec, scores
        else:
            det_rbboxes, det_labels = multiclass_poly_nms_8_points(rbboxes_poly, scores,
                                                                   cfg.score_thr,
                                                                   cfg.nms,
                                                                   max_num=cfg.max_per_img)

            return det_rbboxes, det_labels


    @force_fp32(apply_to=('bbox_preds',))
    def refine_bbox_rbbox(self,
                       rois,
                       labels,
                       bbox_preds,
                       pos_is_gts,
                       img_metas,
                       rec2delta='v2'):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_, rec2delta)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred',))
    def regress_by_class(self,
                         rois,
                         label,
                         bbox_pred,
                         img_meta,
                         rec2delta='v2'):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 5) or (n, 6)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 5)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 5 or rois.size(1) == 6

        if not self.reg_class_agnostic:
            label = label * 5
            inds = torch.stack((label, label + 1, label + 2, label + 3, label + 4), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 5

        if rois.size(1) == 5:
            if rec2delta == 'v1':
                new_rois = delta2rec_v1(bbox_pred, rois, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
            elif rec2delta == 'v2':
                new_rois = delta2rec_v2(bbox_pred, rois, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
                raise AssertionError("rec2delta must be v1/v2")
        else:
            if rec2delta == 'v1':
                bboxes = delta2rec_v1(bbox_pred, rois[:, 1:], self.target_means,
                                  self.target_stds, img_meta['img_shape'])
            elif rec2delta == 'v2':
                bboxes = delta2rec_v2(bbox_pred, rois[:, 1:], self.target_means,
                                  self.target_stds, img_meta['img_shape'])
            else:
                raise AssertionError("rec2delta must be v1/v2")
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois


