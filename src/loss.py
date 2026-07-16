#CG-INN
import torch
import torch.nn as nn


class DWT_HF(nn.Module):
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        hl = -x1 - x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hh = x1 - x2 - x3 + x4
        return torch.cat((hl, lh, hh), 1)


class ICINNLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.dwt_hf = DWT_HF()

    def forward(self, ci, si, di, di_hat_clean,
                di_hat_robust=None, di_hat_wrong=None,
                lambda_r_scale=0.0, lambda_hf=0.0):

        loss_h = self.mse(ci, si)
        loss_hf = self.mse(self.dwt_hf(ci), self.dwt_hf(si))
        loss_c = self.mse(di, di_hat_clean)

        loss_r = torch.tensor(0.0, device=ci.device)
        if di_hat_robust is not None:
            loss_r = 0.5 * self.mse(di, di_hat_robust) + 0.5 * self.l1(di, di_hat_robust)

        loss_s = torch.tensor(0.0, device=ci.device)
        if di_hat_wrong is not None:
            loss_s = self.mse(di_hat_wrong, torch.rand_like(di_hat_wrong))

        total = (self.cfg.LAMBDA_H * loss_h
                 + lambda_hf * loss_hf
                 + self.cfg.LAMBDA_C * loss_c
                 + lambda_r_scale * loss_r
                 + self.cfg.LAMBDA_S * loss_s)

        return total, loss_h, loss_c, loss_r, loss_s, loss_hf
