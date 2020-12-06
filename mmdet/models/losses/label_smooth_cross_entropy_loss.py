import torch.nn as nn
import torch
from ..registry import LOSSES
from .utils import weight_reduce_loss

@LOSSES.register_module
class LabelSmoothCrossEntropyLoss(nn.Module):

    def __init__(self,
                 epsilon=0.1,
                 reduction='mean',
                 loss_weight=1.0):
        super(LabelSmoothCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        reduction = (
            reduction_override if reduction_override else self.reduction)
        with torch.no_grad():
            # num_classes = cls_score.size(1)
            num_classes = 1000
            label = label.clone().detach()
            label_pos = 1. - self.epsilon
            label_neg = self.epsilon / num_classes
            label_one_hot = torch.empty_like(cls_score).fill_(
                label_neg).scatter(1, label.unsqueeze(1), label_pos).detach()

        logs = self.log_softmax(cls_score)
        loss = -torch.sum(logs * label_one_hot, dim=1)

        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
        loss = self.loss_weight * loss
        return loss

