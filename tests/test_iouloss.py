import unittest

import torch
from loss.iou import IoULoss


class TestIoULoss(unittest.TestCase):
    def test_forward(self):
        b, h, w = 2, 5, 5
        score_map = torch.zeros((b, 1, h, w))
        score_map[:, 0, 0:h-1, 0:w-1] = 1
        pred_geo = torch.rand((b, 4, h, w)) * h
        gt_geo = torch.rand((b, 4, h, w)) * h
        l = IoULoss()
        loss = l(pred_geo, gt_geo, mask=score_map)

    def test_forward_when_pred_equals_gt(self):
        b, h, w = 2, 128, 128
        score_map = torch.zeros((b, 1, h, w))
        score_map[:, 0, 15:70, 50:109] = 1
        pred_geo = torch.rand((b, 4, h, w)) * h
        gt_geo = pred_geo.clone()
        l = IoULoss()
        loss = l(pred_geo, gt_geo, mask=score_map)
        self.assertEqual(loss, 0)

    def test_invert_mask(self):
        b, h, w = 3, 10, 10
        score_map = torch.zeros((b, 1, h, w))
        score_map[:, 0, 2:6, 3:7] = 1
        l = IoULoss()
        inverted = l.invert_mask(score_map)
        self.assertEqual((inverted + score_map).sum(), b*h*w)

if __name__ == "__main__":
    pass
