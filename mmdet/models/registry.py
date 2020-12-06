from mmdet.utils import Registry

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
LOSSES = Registry('loss')
DETECTORS = Registry('detector')
# DETECTORS是一个注册表对象，可以将Faster_RCNN等注册到这个对象中
# 即，DETECTORS的module_dict属性中的Faster_RCNN等模型的类
# 实例化Registry类（mmdet/utils/registry.py)，传入的字符串为类名
