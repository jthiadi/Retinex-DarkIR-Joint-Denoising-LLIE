import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim


class FFTLoss(nn.Module):
    """
    Frequency domain loss — helps recover sharp edges
    """

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        return F.l1_loss(
            torch.abs(pred_fft),
            torch.abs(target_fft)
        )


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        l1 = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)

        loss = 0.8 * l1 + 0.2 * mse

        return loss, {
            "l1": l1.item(),
            "mse": mse.item(),
        }