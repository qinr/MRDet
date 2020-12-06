# model settings
model = dict(
    type='DoubleHeadRCNNHBBOBBTransformer4Branch',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientAnchorHead',
        num_classes=2,
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[1.0, 1/2.0, 2.0, 1/3.0, 3.0],
        anchor_strides=[4, 8, 16, 32, 64],
        # anchor_strieds=[2, 4, 8, 16, 32],
        target_means_hbb=[.0, .0, .0, .0],
        target_stds_hbb=[1.0, 1.0, 1.0, 1.0],
        target_means_obb=[0.9, 0, 0, 0.9],
        target_stds_obb=[0.1, 0.1, 0.1, 0.1],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_obb=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRRoIExtractor',
        roi_layer=dict(type='RRoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='DoubleConvFCBBoxHeadOBB4Branch_v2',
        in_channels=256,
        num_convs_xy=2,
        num_convs_wh=4,
        num_fcs_theta=2,
        num_fcs_cls=2,
        xy_conv_out_channels=1024,
        wh_conv_out_channels=1024,
        theta_fc_out_channels=1024,
        cls_fc_out_channels=1024,
        roi_feat_size=7,
        # num_classes=21,
        num_classes=2,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
        reg_class_agnostic=False,
        # loss_cls=dict(
        #     type='LabelSmoothCrossEntropyLoss', epsilon=0.1, loss_weight=1.0),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox_xy=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        loss_bbox_wh=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
        loss_bbox_theta=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerRbbox',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                ignore_iof_thr=-1),
            sampler=dict(
                # type='OHEMSampler_v1',
                type='RandomSamplerRbbox',
                num=512,
                # large_fraction=0.25,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=300,
        max_num=300,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.5, nms=dict(type='poly_nms', iou_thr=0.1), max_per_img=100)
    # soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'HRSC2016DatasetCoco'
data_root = 'data/HRSC2016/'
# data_root = 'data/DOTA_1024_200_1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='MixUp', mixup_ratio=0.5, alpha=1.5, max_bbox_num=500),
    dict(type='RotateAugmentation', rotate_ratio=0.5, small_filter=0),
    # dict(type='Resize', small_filter=6, resize_ratio=1, ratio_range=(0.6, 1), img_scale=(1024, 1024), keep_ratio=True),
    dict(type='Resize',  resize_ratio=1, img_scale=(800, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                    'scale_factor', 'flip', 'img_norm_cfg'))
     #                'mixup', 'mixup_lambd','mixup_num1', 'mixup_num2'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/trainval.json',
        img_prefix=data_root + 'Train/AllImages/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'ImageSets/test.json',
        #          'data/DOTA_1024_200_1.5/test1024/DOTA_test1024.json'],
        #           data_root + 'test1024_ms/DOTA_test1024_ms.json'],
        # ann_file=data_root + 'debug.json',
        img_prefix=data_root + 'Test/AllImages/',
        #            'data/DOTA_1024_200_1.5/test1024/images'],
        #             data_root + 'test1024_ms/images'],
        # img_prefix=data_root + 'test1024_ms/images',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='Step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[80, 110])
    # target_lr=0)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 120
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './work_dirs/dh_faster_rcnn_hbb_obb_transformer_4_branch_r50_fpn_1x/2020042823'
work_dir = './work_dirs/dh_faster_rcnn_hbb_obb_transformer_4_branch_r101_fpn_2x_hrsc2016/2020071921'
load_from = None
resume_from = None
workflow = [('train', 1)]
