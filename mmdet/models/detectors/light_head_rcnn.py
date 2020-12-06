from .base import BaseDetector
from .. import builder
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from ..registry import DETECTORS
from ..plugins import LargeSeperateConv
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
import torch.nn as nn


# @的含义：
# python当解释器读到@这样的修饰符之后，会先解释@后的内容，直接把@下一行的函数或者类作为@后面的函数的参数，然后将返回值赋值给下一行修饰的函数对象
# 这里表明，调用DETECTORS.register_module(FasterRCNN()),将FasterRCNN注册到注册表上
@DETECTORS.register_module
class LightHeadRCNN(BaseDetector, RPNTestMixin, BBoxTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 large_seperate_conv=None):
        super(LightHeadRCNN, self).__init__()
        self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

        if shared_head is not None:
            self.shared_head = builder.build_shared_head(shared_head)

        if rpn_head is not None:
            self.rpn_head = builder.build_head(rpn_head)

        if bbox_head is not None:
            self.bbox_roi_extractor = builder.build_roi_extractor(
                bbox_roi_extractor)
            self.bbox_head = builder.build_head(bbox_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # large_seperate_conv
        if large_seperate_conv is not None:
            self.large_seperate_conv = LargeSeperateConv(**large_seperate_conv)

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_large_seperate_conv(self):
        return hasattr(self, 'large_seperate_conv') and self.large_seperate_conv is not None

    def init_weights(self, pretrained=None):
        super(LightHeadRCNN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_large_seperate_conv:
            self.large_seperate_conv.init_weights()

    def extract_feat(self, img):
        # 提取特征层的最终特征
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img) # x得到的是一个元组，对于resnet，可包含多个conv层的结果
        conv4 = (x[0],)
        conv5 = x[1]

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(conv4) # rpn使用的是resnet的conv4
            # rpn_outs = rpn_cls_score + rpn_bbox_pred
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        large_seperate_conv_output = (self.large_seperate_conv(conv5),)

        # assign gts and sample proposals
        if self.with_bbox :
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # rois是基于特征图尺寸的，不是基于原图尺寸的
            # roi:(roi_batch_inds,x1,y1,x2,y2)
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                large_seperate_conv_output[:self.bbox_roi_extractor.num_inputs], rois)
            # bbox_feats是基于特征图尺寸的，如果需要基于原图尺寸，需要传入scale_factor参数
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)
        conv4 = (x[0],)
        conv5 = x[1]

        proposal_list = self.simple_test_rpn(
            conv4, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        large_seperate_conv_output = (self.large_seperate_conv(conv5),)

        det_bboxes, det_labels = self.simple_test_bboxes(
            large_seperate_conv_output, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        return bbox_results

