import torch

from ..utils import multi_apply
from .transformer_rbbox import (choose_best_match,
                                rec2delta)
from .transformer_obb import (rbboxPoly2Rectangle)


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

    pos_rbboxes_rec = rbboxPoly2Rectangle(pos_rbboxes)
    pos_gt_rbboxes_rec = gt_rbboxes_rec[pos_assigned_gt_inds]
    pos_gt_rbboxes_rec = choose_best_match(pos_rbboxes_rec, pos_gt_rbboxes_rec)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = rec2delta(pos_rbboxes_rec,
                                        pos_gt_rbboxes_rec,
                                        target_means,
                                        target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0

    return labels, label_weights, bbox_targets, bbox_weights

