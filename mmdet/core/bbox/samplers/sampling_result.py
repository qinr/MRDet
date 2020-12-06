import torch

# 包装sampler生成的正负样本
class SamplingResult(object):

    def __init__(self, pos_inds, neg_inds, bboxes, gt_bboxes, assign_result,
                 gt_flags):
        self.pos_inds = pos_inds # 正样本下标
        self.neg_inds = neg_inds # 负样本下标
        self.pos_bboxes = bboxes[pos_inds] # 正样本bbox
        self.neg_bboxes = bboxes[neg_inds] # 负样本bbox
        self.pos_is_gt = gt_flags[pos_inds] # 是gt_bbox的正样本bbox（用于使用gt_bbox补足正样本的情况，是：1，不是：0）

        self.num_gts = gt_bboxes.shape[0] # gt_bbox数目
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1 # 正样本对应的gt_bbox下标
        self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds, :] # 正样本对应的gt_bbox
        if assign_result.labels is not None:
            self.pos_gt_labels = assign_result.labels[pos_inds] # 正样本对应的label
        else:
            self.pos_gt_labels = None

    @property
    def bboxes(self):
        return torch.cat([self.pos_bboxes, self.neg_bboxes])

    @property
    def inds(self):
        return torch.cat([self.pos_inds, self.neg_inds])

    @property
    def bboxes_num(self):
        return len(self.pos_bboxes) + len(self.neg_bboxes)
