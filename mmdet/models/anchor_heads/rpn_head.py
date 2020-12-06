import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
from mmdet.ops import nms
from ..registry import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module
class RPNHead(AnchorHead):

    def __init__(self, in_channels, **kwargs):
        super(RPNHead, self).__init__(2, in_channels, **kwargs)
        # 完成各种属性的赋值
        # 执行_init_layers

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        losses = super(RPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore) # 执行父类的loss方法

        # qr:这里代码不是很严谨 losses可能会返回None,会报错
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    # NMS之前：根据置信度得到nms_pre个anchor
    # 将nms_pre个anchor使用bbox_preds(tx,ty,tw,th)变换得到proposals
    # 对proposals进行NMS，筛去重合度超过nms_thr的proposals
    # NMS之后：根据置信度得到nms_post个proposals
    def get_bboxes_single(self,
                          cls_scores, # 包含不同特征层的结果，以便fpn使用
                          bbox_preds, # cls_scores/bbox_preds/m1v1_anchors的第一维表示哪层特征图
                          mlvl_anchors, # 假设有3层特征图，则cls_scores[0]-bbox_preds[0]-m1v1_anchors[0]对应
                          img_shape, # 同理，cls_scores[1]-bbox_preds[1]-m1v1_anchors[1]对应
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # nms_pre:NMS前，选出置信度前nms_pre高的anchor
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre) # 置信度前nms_pre高的anchor inds
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            # 将选出的anchor由delta变换得到proposals(x1,y1,x2,y2)
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            proposals, _ = nms(proposals, cfg.nms_thr) #根据nms_thr完成NMS
            proposals = proposals[:cfg.nms_post, :] # 选出置信度前nms_post的proposals
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
        # 只返回了(x1,y1,x2,y2,score)，没有返回label，因为RPN是二分类
