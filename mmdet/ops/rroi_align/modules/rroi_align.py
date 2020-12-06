from torch.nn.modules.module import Module
from ..functions.rroi_align import RRoIAlignFunction
from torch.nn.modules.utils import _pair

class RRoIAlign(Module):

    def __init__(self, out_size, spatial_scale, sample_num=0):
        super(RRoIAlign, self).__init__()

        self.out_size = _pair(out_size)
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)

    def forward(self, features, rois):
        return RRoIAlignFunction.apply(features, rois, self.out_size,
                                      self.spatial_scale, self.sample_num)
