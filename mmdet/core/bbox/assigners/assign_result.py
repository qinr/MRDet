import torch

# 用于封装assigner的结果
class AssignResult(object):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts # gt_bbox的数量
        self.gt_inds = gt_inds # 每个bbox对应的gt_bbox的下标，-1：忽略，0：负样本，>0:对应的gt_bbox的下标 + 1
        self.max_overlaps = max_overlaps # 每个bbox与所有gt_bbox的最大overlap
        self.labels = labels

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(self.num_gts), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
