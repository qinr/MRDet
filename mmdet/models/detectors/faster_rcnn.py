from ..registry import DETECTORS
from .two_stage import TwoStageDetector

# @的含义：
# python当解释器读到@这样的修饰符之后，会先解释@后的内容，直接把@下一行的函数或者类作为@后面的函数的参数，然后将返回值赋值给下一行修饰的函数对象
# 这里表明，调用DETECTORS.register_module(FasterRCNN()),将FasterRCNN注册到注册表上
@DETECTORS.register_module
class FasterRCNN(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(FasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
