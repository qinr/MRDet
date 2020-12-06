import torch
import torch.nn as nn
from mmdet.models.registry import HEADS
from mmdet.models import builder
import numpy as np
import copy
from mmdet.core import (rbbox2rroi, rec2delta_v1, rbbox2hbbox, hbbox2rbbox,
                        rbboxPoly2Rectangle_v1, build_assigner, build_sampler,
                        force_fp32, delta2rec_v1, roi2bbox, multi_apply,
                        rbbox2rroi, bbox2roi, bbox_target_rbbox, hbboxList2rbboxRec_v1,
                        choose_best_rroi, hbbox2rbboxRec_v1)



@HEADS.register_module
class RoITransformerHead(nn.Module):
    """
        Args:
            rroi_learner_bbox_roi_extractor：psroi align
            rroi_wrapper_bbox_roi_extractor: rpsroi align
            loss_bbox (dict): Config of localization loss.
            target_means (Iterable): Mean values of regression targets.
            target_stds (Iterable): Std values of regression targets.
            in_channels (int): Number of channels in the input feature map.
            out_dim_reg(int): bbox_reg的输出维度
    """

    def __init__(self,
                 rroi_learner_bbox_roi_extractor,
                 rroi_wrapper_bbox_roi_extractor,
                 bbox_head,
                 in_channels=490,
                 out_dim_reg=5):
        super(RoITransformerHead, self).__init__()
        self.in_channels = in_channels

        if rroi_learner_bbox_roi_extractor is not None:
            self.rroi_learner_bbox_roi_extractor = builder.build_roi_extractor(
                rroi_learner_bbox_roi_extractor)

        if rroi_wrapper_bbox_roi_extractor is not None:
            self.rroi_wrapper_bbox_roi_extractor = builder.build_roi_extractor(
                rroi_wrapper_bbox_roi_extractor)

        self.out_dim_reg = out_dim_reg
        self.bbox_head = builder.build_head(bbox_head)
        ##################################################
        # self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        # self.loss_bbox = builder.build_loss(loss_bbox)
        # self.fc_cls = nn.Linear(in_channels, 2) # 二分类
        # self.loss_cls = builder.build_loss(loss_cls)
        # self.target_means = target_means
        # self.target_stds = target_stds
        ##################################################


    def init_weights(self):
        # nn.init.normal_(self.fc_reg.weight, 0, 0.01)
        # nn.init.constant_(self.fc_reg.bias, 0)
        self.rroi_learner_bbox_roi_extractor.init_weights()
        self.rroi_wrapper_bbox_roi_extractor.init_weights()
        self.bbox_head.init_weights()

    def get_target(self, sampling_results, gt_bboxes_poly, cfg):
        cls_reg_targets = self.bbox_head.get_target_bbox_rbbox(
            sampling_results,
            gt_bboxes_poly,
            cfg)
        # print(bbox_targets)
        return cls_reg_targets


    @force_fp32(apply_to=('bbox_pred', 'cls_score'))
    def loss(self,
             bbox_pred,
             cls_score,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights):
        losses = self.bbox_head.loss(
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights)
        roi_transformer_losses = dict()
        roi_transformer_losses['loss_roi_transformer_cls'] = losses['loss_cls']
        roi_transformer_losses['roi_transformer_acc'] = losses['acc']
        roi_transformer_losses['loss_roi_transformer_bbox'] = losses['loss_bbox']

        return roi_transformer_losses

    def forward_learner(self,
                features,
                proposals):
        '''
        :param feature(tuple(Tensor)): (N, 490, H, W） 
        
        :param proposals(list[Tensor]): [(n,5)], (xmin,ymin,xmax,ymax,scores)，经过RPN之后得到的proposals
        
        :return: 
        '''

        # rroi learner
        assert self.rroi_learner_bbox_roi_extractor is not None
        hrois = bbox2roi(proposals)  # e.g. (roi_batch_ind,xmin,ymin,xmax,ymax)
        x = self.rroi_learner_bbox_roi_extractor(
            features[:self.rroi_learner_bbox_roi_extractor.num_inputs], hrois)  # e.g.(1600,10,7,7)
        cls_score, bbox_pred = self.bbox_head(x)
        # bbox_pred = self.fc_reg(x)  # e.g.(tx,ty,tw,th,ttheta)
        # cls_score = self.fc_cls(x)

        return bbox_pred, cls_score

    def get_rbboxes(self, proposals_list, bbox_pred, pos_is_gt_list, labels,  img_meta):
        assert len(img_meta) == len(proposals_list)
        assert len(img_meta) == len(pos_is_gt_list)

        if self.bbox_head.hbbox2rbbox_poly2rec == 'v1':
            proposals_rec_list = hbboxList2rbboxRec_v1(proposals_list)
        else:
            raise AssertionError("hbbox2rbbox_poly2rec must be v1")
        hrois_rec = rbbox2rroi(proposals_rec_list)
        rrois = self.bbox_head.refine_bbox_rbbox(hrois_rec, labels, bbox_pred, pos_is_gt_list,
                                            img_meta, rec2delta=self.bbox_head.hbbox2rbbox_rec2delta)
        return rrois



    def forward_warpper(self,
                        features,
                        rrois):
        # rroi_wrapper
        assert self.rroi_wrapper_bbox_roi_extractor is not None

        x = self.rroi_wrapper_bbox_roi_extractor(
            features[:self.rroi_wrapper_bbox_roi_extractor.num_inputs], rrois)

        return x


    def simple_test(self, features, proposals, img_meta):
        # rroi_learner
        hrois = bbox2roi(proposals)  # e.g. (roi_batch_ind,xmin,ymin,xmax,ymax)
        x1 = self.rroi_learner_bbox_roi_extractor(
            features[:self.rroi_learner_bbox_roi_extractor.num_inputs], hrois)  # e.g.(1600,10,7,7)
        x1 = x1.view(x1.size(0), -1)  # e.g. (1600,490)
        cls_score, bbox_pred = self.bbox_head(x1)  # e.g.(tx,ty,tw,th,ttheta)

        bbox_label = cls_score.argmax(dim=1)

        if self.bbox_head.hbbox2rbbox_poly2rec == 'v1':
            proposals_rec = hbboxList2rbboxRec_v1(proposals)
        else:
            raise AssertionError("poly2rec must be v1")
        hrois_rec = rbbox2rroi(proposals_rec)

        rrois = self.bbox_head.regress_by_class(hrois_rec, bbox_label, bbox_pred, img_meta, self.bbox_head.hbbox2rbbox_rec2delta)

        rrois_enlarge = copy.deepcopy(rrois)
        rrois_enlarge[:, 3] = rrois_enlarge[:, 3] * self.rroi_wrapper_bbox_roi_extractor.w_enlarge
        rrois_enlarge[:, 4] = rrois_enlarge[:, 4] * self.rroi_wrapper_bbox_roi_extractor.h_enlarge

        # rroi_wrapper
        x2 = self.rroi_wrapper_bbox_roi_extractor(
            features[:self.rroi_wrapper_bbox_roi_extractor.num_inputs], rrois_enlarge)
        return x2, rrois








