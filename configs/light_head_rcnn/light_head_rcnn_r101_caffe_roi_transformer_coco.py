# model settings
model = dict(
    type='LightHeadRCNNOBB',
    pretrained='open-mmlab://resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4, # resnet的stage数目
        strides=(1, 2, 2, 1),
        dilations=(1, 1, 1, 2),
        out_indices=(2, 3), # backbone返回的特征层，可不只返回最终的特征层，也可返回中间结果
        frozen_stages=1, # 冻结的stage数量，即该stage不更新参数，-1表示所有的stage都更新参数
        style='caffe'),# 网络风格：如果设置pytorch，则stride为2的层是conv3x3的卷积层；如果设置caffe，则stride为2的层是第一个conv1x1的卷积层
    # large_seperate_conv
    large_seperate_conv=dict(
        in_channels=2048,
        mid_channels=256,
        out_channels=10*7*7,
        k=15,
        mode=0),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024, # 使用的是CONV4
        feat_channels=512,
        anchor_scales=[2, 4, 8, 16, 32],
        # anchor_scales=[4, 8, 16, 32, 64], # 扩大anchor scales
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16], # 16的原因是经过resnet，原图尺度变为了原来的1/16
        target_means=[.0, .0, .0, .0], # delta的均值和方差
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_transformer_head=dict(
        type='RoITransformerHead',
        in_channels=490,
        rroi_learner_bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='PSRoIAlign', pooled_size=7, group_size=7, sample_num=2),
            out_channels=10,
            featmap_strides=[16]),
        rroi_wrapper_bbox_roi_extractor=dict(
            type='SingleRRoIExtractor',
            roi_layer=dict(type='RPSRoIAlign', pooled_size=7, group_size=7, sample_num=2),
            out_channels=10,
            featmap_strides=[16]),
        bbox_head=dict(
            type='SharedFCBBoxHeadOBB',
            hbbox2rbbox_poly2rec='v2',
            hbbox2rbbox_rec2delta='v2',
            num_fcs=1,
            fc_out_channels=2048,
            roi_feat_size=7,
            in_channels=10,
            # num_classes=81,
            num_classes=16,
            target_means=[0., 0., 0., 0., 0,],
            target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
            # reg_class_agnostic=False,
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), # use_sigmoid：True使用sigmoid分类，False使用softmax分类
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    bbox_head=dict(
        type='SharedFCBBoxHeadOBB',
        rbbox2rbbox_poly2rec='v2',
        rbbox2rbbox_rec2delta='v2',
        num_fcs=1,
        fc_out_channels=2048,
        roi_feat_size=7,
        in_channels=10,
        # num_classes=81,
        num_classes=16,
        target_means=[0., 0., 0., 0., 0,],
        target_stds=[0.05, 0.05, 0.1, 0.1, 0.05],
        # reg_class_agnostic=False,
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0), # use_sigmoid：True使用sigmoid分类，False使用softmax分类
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        # assigner用于完成anchor和gt_bbox匹配
        assigner=dict(
            type='MaxIoUAssigner', # 将bbox与gt_bbox匹配，生成正负样本
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3, # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1), # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        # sampler完成正负样本生成
        sampler=dict(
            type='RandomSampler', # 正负样本提取器
            # num=512, # rpn_batch_size 需提取的正负样本数量
            num=256,
            pos_fraction=0.5, # 正样本比例
            neg_pos_ub=-1, # 最大负正样本比例，大于该比例的负样本忽略，-1表示不忽略
            add_gt_as_proposals=False), # 把ground truth加入proposal作为正样本
        allowed_border=0, # 允许在bbox周围外扩一定的像素
        pos_weight=-1, # 正样本权重，-1表示不改变原始的权重
        debug=False),
    # rpn_proposal:完成NMS
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    roi_transformer=dict(
        assigner=dict(
            type='MaxIoUAssigner',  # 将bbox与gt_bbox匹配，生成正负样本
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,  # 正样本的iou最小值。如果assign给ground truth的anchors中最大的IOU低于0.3，则忽略所有的anchors，否则保留最大IOU的anchor
            ignore_iof_thr=-1),
        sampler = dict(
            type='RandomSampler',
            num=512, # num=800
            pos_fraction=0.25, # 保证所有的正样本都可以参与训练
            neg_pos_ub=-1,
            add_gt_as_proposals=True), # 防止没有正样本
        pos_weight=-1, # 忽略bbox的阈值，当ground truth中包含需要忽略的bbox时使用，-1表示不忽略
        debug = False),  # 从512个中随机选择288个填补（存在重复的），保证出来的结果是800个
    rcnn=dict(
        assigner=dict(
            type='MaxPolyIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomRotationSampler',
            num=512,
            pos_fraction=0.25,
            # pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='poly_nms', iou_thr=0.1), max_per_img=1000)
# soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
# dataset_type = 'CocoDataset'
dataset_type = 'DOTADatasetCoco'
# data_root = '/data/qrrr/mmdetection/data/coco/'
data_root = 'data/DOTACoco_small/'
img_norm_cfg = dict(
    #mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# 输入图像初始化，减去均值mean并除以方差std，to_rgb表示将bgr转为rgb
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='RotateAugmentation', rotate_ratio=0, small_filter=6),
    dict(type='Resize', img_scale=(1024,1024), keep_ratio=True), # 输入图像尺寸，最大边1200，最小边800，keep_ratio：是否保持原图的长宽比
    dict(type='RandomFlip', flip_ratio=0.5),  # 随机翻转，flip_ratio表示翻转可能性
    dict(type='Normalize', **img_norm_cfg), # 图像初始化参数，标准化
    dict(type='Pad', size_divisor=32), # 对图像进行resize时的最小单位，32表示所有的图像都会被resize成32的倍数
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
# 以下这种配置方法是在test阶段不采用数据增强的方法
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False, # 不采用flip数据增强
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
    imgs_per_gpu=2, # 每个gpu计算的图像数量
    workers_per_gpu=2, # 每个gpu分配的线程数
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # img_prefix=data_root + 'train2017/',
	    # ann_file=data_root + 'trainval.txt',
        # DOTA的37373份数据中存在不含gt_bbox的数据，这种图片是不存在的，因此不参与训练
	    # 只有含gt_bbox的图片才会参与训练
	    # img_prefix=data_root,
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        # ann_file=data_root + 'val1024/' + 'val.txt',
	    # img_prefix=data_root + 'val1024/',
        ann_file=data_root + 'trainval1024/DOTA_trainval1024.json',
        img_prefix=data_root + 'trainval1024/images',
	    pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instances_val2017.json',
        # img_prefix=data_root + 'val2017/',
        # ann_file=data_root + 'val1024/' + 'val.txt',
	    # img_prefix=data_root + 'val1024/',
        # ann_file=data_root + 'test.txt',
        # img_prefix=data_root,
        ann_file=data_root + 'test1024/DOTA_test1024.json',
        img_prefix=data_root + 'test1024/images',
	    pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# 1 image: lr = 0.0005 (roi transformer论文配置）
# 这里的lr跟batch_size是线性关系的，默认lr=0.02对应8gpu*2images/gpu，则4gpu*2images/gpu对应的lr是0.01
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))  # 梯度均衡参数
# learning policy
lr_config = dict(
    policy='step', # 优化策略
    warmup='linear', # 初始的学习率增加的策略，linear为线性增加
    warmup_iters=500, # 前500次迭代中学习率逐渐增加
    warmup_ratio=1.0 / 3, # warmup的增长速率
    step=[8, 11]) # 在第几个epoch降低学习率
checkpoint_config = dict(interval=1) # 1个epoch记录一次checkpoint
# yapf:disable
log_config = dict(
    interval=50, # 50个step记录一次log
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/light_head_rcnn_r101_caffe_roi_transformer'
load_from = None
resume_from = None
workflow = [('train', 1)]