import torch

from torch import nn
from torch.nn.modules.loss import MarginRankingLoss
from loss.iou import IoULoss
from loss.angle import AngleLoss


class FOTSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # pixel wise classification loss for the score map
        self.l_cls = nn.BCELoss()
        self.l_iou = IoULoss()
        self.l_angle = AngleLoss()

    def forward(self, pred_score, pred_geo, gt_score, gt_geo):
        pred_loc, gt_loc = pred_geo[:, :4, :, :], gt_geo[:, :4, :, :]
        pred_angle, gt_angle = pred_geo[:, 4, :, :], gt_geo[:, 4, :, :]
        l_cls = self.l_cls(pred_score, gt_score)
        l_iou = self.l_iou(pred_loc, gt_loc, mask=gt_score)
        l_angle = self.l_angle(pred_angle, gt_angle, mask=gt_score)
        l_reg = l_cls + l_iou + l_angle
        return l_reg
