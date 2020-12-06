# model settings
model = dict(
    type='FasterRCNN',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
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
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[1.0, 1 / 2.0, 2.0, 1 / 3.0, 3.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0], # delta的均值和方差
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIPool', out_size=7),
        # roi_layer=dict(type='RoIPool', out_size=7),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=2,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        # assigner用于完成anchor和gt_bbox匹配
        assigner=dict(
            type='MaxIoUAssigner', # 将bbox与gt_bbox匹配，生成正负样本
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,# 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1), # # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        # sampler完成正负样本生成
        sampler=dict(
            type='RandomSampler', # 正负样本提取器
            num=256, # 需提取的正负样本数量
            pos_fraction=0.5, # 正样本比例
            neg_pos_ub=-1, # 最大负正样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False), # 把ground truth加入proposal作为正样本
        allowed_border=0, # 允许在bbox周围外扩一定的像素
        pos_weight=-1,  # 正样本权重，-1表示不改变原始的权重
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
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
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
        score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100))
# dataset settings
# dataset_type = 'CocoDataset'
dataset_type = 'HRSC2016DatasetVOCH'
# data_root = 'data/coco/'
data_root = 'data/HRSC2016/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# 输入图像初始化，减去均值mean并除以方差std，to_rgb表示将bgr转为rgb
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', resize_ratio=1, img_scale=(800, 512), keep_ratio=True),  # 输入图像尺寸，最大边1333，最小边800，keep_ratio：是否保持原图的长宽比
    dict(type='RandomFlip', flip_ratio=0.5), # 随机翻转，flip_ratio表示翻转可能性
    dict(type='Normalize', **img_norm_cfg), # 图像初始化参数，标准化
    dict(type='Pad', size_divisor=32), # 对图像进行resize时的最小单位，32表示所有的图像都会被resize成32的倍数
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 512),
        flip=False, # 是否要采用flip数据增强
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
############## modify by qr ##############
data = dict(
    imgs_per_gpu=2, # 每个gpu计算的图像数量
    workers_per_gpu=2, # 每个gpu分配的线程数
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # img_prefix=data_root + 'train2017/',
	    ann_file=data_root + 'ImageSets/random_sample/trainval_30.txt',
	    img_prefix=data_root + 'FullDataSet/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        ann_file=data_root + 'ImageSets/random_sample/test.txt',
        img_prefix=data_root + 'FullDataSet/',
	    pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        ann_file=data_root + 'ImageSets/random_sample/test.txt',
	    img_prefix=data_root + 'FullDataSet/',
	    pipeline=test_pipeline))
############################################
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 24
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/al_detection/faster_rcnn_r50_fpn_hrsc2016/30_coco/'
# load_from = None
load_from = './checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15_classes_2.pth'
resume_from = None
workflow = [('train', 1)]
