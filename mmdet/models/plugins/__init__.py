from .generalized_attention import GeneralizedAttention
from .non_local import NonLocal2D
from .large_seperate_conv import LargeSeperateConv
from .center_pooling import CenterPooling
from .se_attention import SEAttention
from .extreme_pooling import ExtremePooling
from .center_pooling_v1 import CenterPoolingV1

__all__ = ['NonLocal2D', 'GeneralizedAttention', 'LargeSeperateConv',
           'CenterPooling', 'SEAttention', 'ExtremePooling', 'CenterPoolingV1']
# add LargeSeperateConv/CenterPooling/SEAttention by qr
