import torch
from torch import nn


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll = nn.NLLLoss()

    @staticmethod
    def invert_mask(mask):
        max = mask.max()
        mask = - (mask - max) / max
        return mask

    def forward(self, pred_loc, gt_loc, mask=None):
        if mask is not None:
            pred_loc = mask * pred_loc
            gt_loc = mask * gt_loc
        xt, xb, xl, xr = torch.split(pred_loc, split_size_or_sections=1, dim=1)
        xt_, xb_, xl_, xr_ = torch.split(gt_loc, split_size_or_sections=1, dim=1)
        X = (xt + xb) * (xl + xr)
        X_ = (xt_ + xb_) * (xl_ + xr_)
        Ih = torch.min(xt, xt_, ) + torch.min(xb, xb_)
        Iw = torch.min(xl, xl_, ) + torch.min(xr, xr_)
        I = Ih * Iw
        inverted_mask = self.invert_mask(mask)
        U = X + X_ - I
        I = I + inverted_mask
        U = U + inverted_mask
        IoU = I/U
        # target = torch.ones_like(IoU)
        return (-torch.log(IoU)).mean()
