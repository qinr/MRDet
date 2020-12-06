from torch import nn

from mmdet.utils import build_from_cfg
from .registry import (BACKBONES, DETECTORS, HEADS, LOSSES, NECKS,
                       ROI_EXTRACTORS, SHARED_HEADS)

# tools/train.py中使用build_detector建立模型时实际调用的函数
def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ] # 构建多个模型
        return nn.Sequential(*modules) # 将多个模型组合
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_roi_extractor(cfg):
    return build(cfg, ROI_EXTRACTORS)


def build_shared_head(cfg):
    return build(cfg, SHARED_HEADS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    # 将模型注册到注册表中
    # DETECTORS为mmdet.models.registry下的类
    return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
