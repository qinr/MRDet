import torch

from ..utils import multi_apply
from .transformer_rbbox import (choose_best_match, rec2delta_v1, get_best_begin_point,
                                rbboxPoly2Rectangle_v1, rec2delta_best_match_v1,
                                rec2delta_v2, hbbox2rbboxRec_v1)
from .transformer_obb import (rbboxPoly2Rectangle_v2, hbbox2rbboxRec_v2)


def bbox_target_rbbox(pos_bboxes_list,
                neg_bboxes_list,
                pos_gt_labels_list,
                pos_assigned_gt_inds_list,
                gt_bboxes_poly_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        bbox_target_rbbox_single,
        pos_bboxes_list,
        neg_bboxes_list,
        pos_gt_labels_list,
        pos_assigned_gt_inds_list,
        gt_bboxes_poly_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def bbox_target_rbbox_single(pos_bboxes,
                       neg_bboxes,
                       pos_gt_labels,
                       pos_assigned_gt_inds,
                       gt_rbboxes_poly, # 带方向
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_bboxes.size(0)
    num_neg = neg_bboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_bboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_bboxes.new_zeros(num_samples)
    bbox_targets = pos_bboxes.new_zeros(num_samples, 5)
    bbox_weights = pos_bboxes.new_zeros(num_samples, 5)

    pos_gt_rbboxes_poly = gt_rbboxes_poly[pos_assigned_gt_inds] # 带方向
    # 将gt_bboxes_poly的点的顺序调整成与(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax)最接近的
    # 由于rpn得到的proposals是(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax)的顺序的
    # 两者之间的角度差距小，比较容易回归

    # pos_gt_rbboxes_rec = rbboxPoly2Rectangle_v2(pos_gt_rbboxes_poly)
    pos_gt_rbboxes_rec = rbboxPoly2Rectangle_v1(pos_gt_rbboxes_poly)
    # pos_hbboxes_rec = hbbox2rbboxRec_v2(pos_bboxes)
    pos_hbboxes_rec = hbbox2rbboxRec_v1(pos_bboxes)

    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = rec2delta_v2(pos_hbboxes_rec,
                                         pos_gt_rbboxes_rec,
                                         target_means,
                                         target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    # labels：前num_pos个是正样本对应的gt_labels，后面负样本的对应的是0
    # label_weights:每个样本分类的权重，都是1
    # bbox_targets:前num_pos个是正样本与其对应的gt box的delta，负样本是0
    # bbox_weights:每个样本回归的权重，正样本是1，负样本是0，负样本不参与回归
    return labels, label_weights, bbox_targets, bbox_weights


def rbbox_target_rbbox(pos_rbboxes_list,
                neg_rbboxes_list,
                pos_assigned_gt_inds_list,
                gt_rbboxes_rec_list,
                pos_gt_labels_list,
                cfg,
                reg_classes=1,
                target_means=[.0, .0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
                concat=True):
    labels, label_weights, bbox_targets, bbox_weights = multi_apply(
        rbbox_target_rbbox_single,
        pos_rbboxes_list,
        neg_rbboxes_list,
        pos_assigned_gt_inds_list,
        gt_rbboxes_rec_list,
        pos_gt_labels_list,
        cfg=cfg,
        reg_classes=reg_classes,
        target_means=target_means,
        target_stds=target_stds)

    if concat:
        labels = torch.cat(labels, 0)
        label_weights = torch.cat(label_weights, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        bbox_weights = torch.cat(bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def rbbox_target_rbbox_single(pos_rbboxes,
                       neg_rbboxes,
                       pos_assigned_gt_inds, # 带方向
                       gt_rbboxes_rec,
                       pos_gt_labels,
                       cfg,
                       reg_classes=1,
                       target_means=[.0, .0, .0, .0, .0],
                       target_stds=[1.0, 1.0, 1.0, 1.0, 1.0]):
    num_pos = pos_rbboxes.size(0)
    num_neg = neg_rbboxes.size(0)
    num_samples = num_pos + num_neg
    labels = pos_rbboxes.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_rbboxes.new_zeros(num_samples)
    bbox_targets = pos_rbboxes.new_zeros(num_samples, 5)
    bbox_weights = pos_rbboxes.new_zeros(num_samples, 5)

    pos_rbboxes_rec = rbboxPoly2Rectangle_v2(pos_rbboxes)
    pos_gt_rbboxes_rec = gt_rbboxes_rec[pos_assigned_gt_inds]
    pos_gt_rbboxes_rec = choose_best_match(pos_rbboxes_rec, pos_gt_rbboxes_rec)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        # 目标pos_gt_bboxes和pos_bboxes满足w>h的条件
        # 表示两者的方向要么差不多一致，要么相差近180
        # 调整pos_gt_bboxes的方向，使其与pos_bboxes满足差不多一致的状态，便于回归
        pos_bbox_targets = rec2delta_v1(pos_rbboxes_rec,
                                        pos_gt_rbboxes_rec,
                                        target_means,
                                        target_stds)
        # pos_bbox_targets = rec2delta_v2(pos_rbboxes_rec,
        #                                 pos_gt_rbboxes_rec,
        #                                 target_means,
        #                                 target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights

