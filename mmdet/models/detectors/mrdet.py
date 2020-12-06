import torch
import torch.nn as nn
import copy
from mmdet.core import (bbox2roi, build_assigner, build_sampler,
                        mask_2_rbbox_list, ndarray2tensor, rbbox2result,
                        enlarge_bridge, get_best_begin_point_list, rbboxPoly2RectangleList_v2,
                        rbboxPoly2Rectangle_v2, rbboxPoly2rroiRec,
                        bbox_mapping, merge_aug_rotate_bboxes,
                        multiclass_poly_nms_8_points)
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import RPNTestMixin, BBoxTestMixin
from .. import builder


@DETECTORS.register_module
class MRDet(BaseDetector, RPNTestMixin, BBoxTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 shared_head=None,
                 rpn_head=None,
                 shared_head_rbbox=None,
                 rbbox_roi_extractor=None,
                 rbbox_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(MRDet, self).__init__()
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

        if shared_head_rbbox is not None:
            self.shared_head_rbbox = builder.build_shared_head(shared_head_rbbox)

        if rbbox_head is not None:
            self.rbbox_roi_extractor = builder.build_roi_extractor(
                rbbox_roi_extractor)
            self.rbbox_head = builder.build_head(rbbox_head)

        if mask_head is not None:
            if mask_roi_extractor is not None:
                self.mask_roi_extractor = builder.build_roi_extractor(
                    mask_roi_extractor)
                self.share_roi_extractor = False
            else:
                self.share_roi_extractor = True
                self.mask_roi_extractor = self.bbox_roi_extractor
            self.mask_head = builder.build_head(mask_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_shared_head_rbbox(self):
        return hasattr(self, 'shared_head_rbbox') and self.shared_head_rbbox is not None

    @property
    def with_rbbox(self):
        return hasattr(self, 'rbbox_head') and self.rbbox_head is not None


    def init_weights(self, pretrained=None):
        super(MRDet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_shared_head_rbbox:
            self.shared_head_rbbox.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_rbbox:
            self.rbbox_roi_extractor.init_weights()
            self.rbbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, img):
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).cuda()
        # bbox head
        rois = bbox2roi([proposals])
        bbox_cls_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        bbox_reg_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs],
            rois,
            roi_scale_factor=self.reg_roi_scale_factor)
        if self.with_shared_head:
            bbox_cls_feats = self.shared_head(bbox_cls_feats)
            bbox_reg_feats = self.shared_head(bbox_reg_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)
        outs += (cls_score, bbox_pred)
        return outs

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
        # torch.cuda.empty_cache()
        # if img_meta[0]['mixup']:
        #     pass
        # if len(gt_bboxes[0]) > 500:
        #     torch.cuda.empty_cache()
        x = self.extract_feat(img)


        gt_rbboxes_poly = mask_2_rbbox_list(gt_masks)  # list(ndarray)
        gt_rbboxes_poly = ndarray2tensor(gt_rbboxes_poly, gt_bboxes[0].device)
        # gt_bboxes = rbbox2hbbox(gt_rbboxes_poly)

        gt_rbboxes_poly = get_best_begin_point_list(gt_rbboxes_poly)
        # gt_rbboxes_poly = enlarge_bridge(gt_rbboxes_poly, gt_labels, 1.2, 1.6)
        # gt_bboxes = rbbox2hbbox(gt_rbboxes_poly)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, gt_rbboxes_poly,
                                          None,
                                          img_meta,
                                          self.train_cfg.rpn)
            try:
                # torch.cuda.empty_cache()
                rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    import pdb
                    pdb.set_trace()
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    rpn_losses = self.rpn_head.loss(
                        *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_rotate_list = self.rpn_head.get_bboxes(*proposal_inputs)
            # 经过FPN之后有多层特征图，针对每一层特征图得到对应的anchor+proposals（大小基于原图）
            # 将所有层的proposals汇总，做NMS，得到proposal_list
        else:
            proposal_rotate_list = proposals

        # assign gts and sample proposals
        if self.with_bbox:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_rotate_list[i],
                                                     gt_rbboxes_poly[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_rotate_list[i],
                    gt_rbboxes_poly[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        if self.with_bbox:
            rois = rbboxPoly2rroiRec([res.bboxes for res in sampling_results])
            # rois[:, 3] = rois[:, 3] * self.bbox_roi_extractor.w_enlarge
            # rois[:, 4] = rois[:, 4] * self.bbox_roi_extractor.h_enlarge
            bbox_cls_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs],
                                                  rois)
            # bbox_cls_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs],
            #                                          rois,
            #                                          roi_w_scale_factor=self.bbox_roi_extractor.w_enlarge,
            #                                          roi_h_scale_factor=self.bbox_roi_extractor.h_enlarge)
            bbox_reg_feats = bbox_cls_feats
            if self.with_shared_head:
                bbox_cls_feats = self.shared_head(bbox_cls_feats)
                bbox_reg_feats = self.shared_head(bbox_reg_feats)
            cls_score, bbox_xy_pred, bbox_wh_pred, bbox_theta_pred = self.bbox_head(bbox_cls_feats,
                                                  bbox_reg_feats)
            gt_rbboxes_rec = rbboxPoly2RectangleList_v2(gt_rbboxes_poly)
            bbox_targets = self.bbox_head.get_target_rbbox2rbbox(sampling_results,
                                                     gt_rbboxes_rec,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score,
                                            bbox_xy_pred,
                                            bbox_wh_pred,
                                            bbox_theta_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        if self.with_mask:
            pass

        return losses

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_rotate_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        if self.with_bbox:
            rois = rbboxPoly2rroiRec(proposal_rotate_list)
            # rois_enlarge = copy.deepcopy(rois)
            # rois_enlarge[:, 3] = rois_enlarge[:, 3] * self.bbox_roi_extractor.w_enlarge
            # rois_enlarge[:, 4] = rois_enlarge[:, 4] * self.bbox_roi_extractor.h_enlarge
            bbox_cls_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs],
                                                     rois)
            # bbox_cls_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs],
            #                                          rois,
            #                                          roi_w_scale_factor=self.bbox_roi_extractor.w_enlarge,
            #                                          roi_h_scale_factor=self.bbox_roi_extractor.h_enlarge)
            bbox_reg_feats = bbox_cls_feats
            if self.with_shared_head:
                bbox_cls_feats = self.shared_head(bbox_cls_feats)
                bbox_reg_feats = self.shared_head(bbox_reg_feats)
            cls_score, bbox_xy_pred, bbox_wh_pred, bbox_theta_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

            img_shape=img_meta[0]['ori_shape']
            scale_factor = img_meta[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_det_rbbox2rbbox(
                rois,
                cls_score,
                bbox_xy_pred,
                bbox_wh_pred,
                bbox_theta_pred,
                img_shape,
                scale_factor,
                rescale=True,
                cfg=self.test_cfg.rcnn)
            rbbox_results = rbbox2result(det_bboxes, det_labels,
                                    self.bbox_head.num_classes)

        if not self.with_mask:
            return rbbox_results
        else:
            return

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rotate_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(self.extract_feats(imgs), img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            # TODO more flexible
            proposals = [bbox_mapping(proposal_list[0][:, :8], img_shape,
                                     scale_factor, flip)]
            rois = rbboxPoly2rroiRec(proposals)
            # rois_enlarge = copy.deepcopy(rois)
            # rois_enlarge[:, 3] = rois_enlarge[:, 3] * self.bbox_roi_extractor.w_enlarge
            # rois_enlarge[:, 4] = rois_enlarge[:, 4] * self.bbox_roi_extractor.h_enlarge
            bbox_cls_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs],
                                                     rois,
                                                     roi_w_scale_factor=self.bbox_roi_extractor.w_enlarge,
                                                     roi_h_scale_factor=self.bbox_roi_extractor.h_enlarge)
            bbox_reg_feats = bbox_cls_feats
            if self.with_shared_head:
                bbox_cls_feats = self.shared_head(bbox_cls_feats)
                bbox_reg_feats = self.shared_head(bbox_reg_feats)
            cls_score, bbox_xy_pred, bbox_wh_pred, bbox_theta_pred = self.bbox_head(bbox_cls_feats, bbox_reg_feats)

            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            bboxes, scores = self.bbox_head.get_det_rbbox2rbbox(
                rois,
                cls_score,
                bbox_xy_pred,
                bbox_wh_pred,
                bbox_theta_pred,
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        merged_bboxes, merged_scores = merge_aug_rotate_bboxes(
            aug_bboxes, aug_scores, img_metas, self.test_cfg.rcnn)
        det_bboxes, det_labels = multiclass_poly_nms_8_points(merged_bboxes,
                                                              merged_scores,
                                                              self.test_cfg.rcnn.score_thr,
                                                              self.test_cfg.rcnn.nms,
                                                              max_num=self.test_cfg.rcnn.max_per_img)


        bbox_results = rbbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            pass
        else:
            return bbox_results
