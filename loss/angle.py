import torch
from torch import nn


class AngleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll = nn.NLLLoss()

    def forward(self, pred_angle, gt_angle, mask=None):
        if mask is not None:
            pred_angle = mask * pred_angle
            gt_angle = mask * gt_angle
        # CHECK: Loss is ignored for non-text regions 
        return (1 - torch.cos(pred_angle - gt_angle)).mean()
