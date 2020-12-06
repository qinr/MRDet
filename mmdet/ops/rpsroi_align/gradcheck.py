import torch
from torch.autograd import gradcheck

import os.path as osp
import sys
sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from rpsroi_align import RPSRoIAlign  # noqa: E402
import math

PI = 3.1415926
# print(math.sin(PI/2))

feat = torch.randn(4, 64, 15, 15, requires_grad=True).cuda()
rois = torch.Tensor([[0, 25, 25, 50, 50, PI / 6], [0, 40, 30, 25, 55, PI / 4],
                     [1, 87, 80, 43, 80, PI / 2], [1, 90, 50, 17, 33, PI * 6 / 7]]).cuda()
inputs = (feat, rois)
print('Gradcheck for rpsroi align...')
test = gradcheck(RPSRoIAlign(4, 1.0 / 8, 4, 2), inputs, eps=1e-5, atol=1e-2)
print(test)
