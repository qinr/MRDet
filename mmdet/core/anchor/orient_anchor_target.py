import torch

from ..bbox import (PseudoSampler, assign_and_sample, bbox2delta, build_assigner,
                    delta2hbboxrec5, hbbox2rbboxRec_v2, rbboxPoly2Rectangle,
                    rec2target)
from ..utils import multi_apply


def orient_anchor_target(bbox_pred_list,
                         anchor_list,
                         valid_flag_list,
                         gt_bboxes_list,
                         gt_rbboxes_poly_list,
                         img_metas,
                         target_means_hbb,
                         target_stds_hbb,
                         target_means_obb,
                         target_stds_obb,
                         cfg,
                         gt_bboxes_ignore_list=None,
                         gt_labels_list=None,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    bbox_pred_new_list = []
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
        bbox_preds = []
        for j in range(len(bbox_pred_list)):
            bbox_preds.append(bbox_pred_list[j][i].permute(1, 2, 0).reshape(-1, 4))
        bbox_preds = torch.cat(bbox_preds)
        bbox_pred_new_list.append(bbox_preds)


    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     all_obb_targets, all_obb_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         orient_anchor_target_single,
         bbox_pred_new_list,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_rbboxes_poly_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         target_means_hbb=target_means_hbb,
         target_stds_hbb=target_stds_hbb,
         target_means_obb=target_means_obb,
         target_stds_obb=target_stds_obb,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    obb_targets_list = images_to_levels(all_obb_targets, num_level_anchors)
    obb_weights_list = images_to_levels(all_obb_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, obb_targets_list, obb_weights_list,
            num_total_pos, num_total_neg)


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def orient_anchor_target_single(bbox_pred,
                                flat_anchors,
                                valid_flags,
                                gt_bboxes,
                                gt_rbboxes_poly,
                                gt_bboxes_ignore,
                                gt_labels,
                                img_meta,
                                target_means_hbb,
                                target_stds_hbb,
                                target_means_obb,
                                target_stds_obb,
                                cfg,
                                label_channels=1,
                                sampling=True,
                                unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    # inside_flags: 返回在图中的anchor对应的索引
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]
    bbox_pred = bbox_pred[inside_flags, :]
    # 筛选后在图中的anchor

    # 将anchor和gt_bbox匹配，得到正样本和负样本, 并用sampler将这些结果进行封装，方便之后使用
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
    obb_targets = torch.zeros_like(anchors)
    obb_weights = torch.zeros_like(anchors)

    pos_inds = sampling_result.pos_inds # 正样本索引
    neg_inds = sampling_result.neg_inds # 负样本索引
    pos_bbox_pred = bbox_pred[pos_inds, :]
    if len(pos_inds) > 0:
        pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                      sampling_result.pos_gt_bboxes,
                                      target_means_hbb, target_stds_hbb)
        # 将bbox转化为delta，并使用target_means,target_stds标准化
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0 # 正样本权重是1，负样本权重是0
        pos_bbox_rec = delta2hbboxrec5(sampling_result.pos_bboxes,
                              pos_bbox_pred,
                              target_means_hbb,
                              target_stds_hbb)
        pos_gt_rbboxes_poly = gt_rbboxes_poly[sampling_result.pos_assigned_gt_inds, :]
        pos_gt_rbboxes_rec = rbboxPoly2Rectangle(pos_gt_rbboxes_poly)
        pos_obb_targets = rec2target(pos_bbox_rec, pos_gt_rbboxes_rec,
                                        target_means_obb, target_stds_obb)
        obb_targets[pos_inds, :] = pos_obb_targets
        obb_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        obb_targets = unmap(obb_targets, num_total_anchors, inside_flags)
        obb_weights = unmap(obb_weights, num_total_anchors, inside_flags)
    # labels:每个anchor对应的label
    # label_weights:每个anchor cls_loss的权重，负样本权重为1，正样本权重可为1也可为其他值
    # bbox_targets：每个anchor与其对应的gt_bbox之前的delta，用于回归
    # bbox_weights: 每个anchor bbox_reg的权重，正样本为1，负样本为0
    # pos_inds：anchor中正样本的索引
    # neg_inds: anchor中负样本的索引
    return (labels, label_weights, bbox_targets, bbox_weights, obb_targets,
            obb_weights, pos_inds, neg_inds)

# 判断anchor是否超出图片边界
def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


