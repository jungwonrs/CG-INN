import torch
import torch.nn as nn
import numpy as np


def _quality_to_factor(q):
    return 5000.0 / q if q < 50 else 200.0 - q * 2.0


Y_TABLE = torch.tensor([
    [16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],[14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99],
], dtype=torch.float32)

C_TABLE = torch.tensor([
    [17,18,24,47,99,99,99,99],[18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],[47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],[99,99,99,99,99,99,99,99],
], dtype=torch.float32)


def _quant_table(quality, comp='y'):
    base = Y_TABLE if comp == 'y' else C_TABLE
    f = _quality_to_factor(quality)
    t = torch.floor((base * f + 50.0) / 100.0)
    return torch.clamp(t, 1.0, 255.0)


def _dct_matrix():
    m = torch.zeros(8, 8)
    for i in range(8):
        for j in range(8):
            if i == 0:
                m[i, j] = 1.0 / np.sqrt(8.0)
            else:
                m[i, j] = np.sqrt(2.0 / 8.0) * np.cos((2*j + 1) * i * np.pi / 16.0)
    return m


class DCT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mat', _dct_matrix())

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H//8, 8, W//8, 8).permute(0, 1, 2, 4, 3, 5)
        return torch.matmul(self.mat, torch.matmul(x, self.mat.t()))


class IDCT(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mat', _dct_matrix().t())

    def forward(self, x, H, W):
        x = torch.matmul(self.mat, torch.matmul(x, self.mat.t()))
        B, C = x.shape[:2]
        return x.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)


class _DiffRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class RGBToYCbCr(nn.Module):
    def forward(self, x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y  =  0.299*r + 0.587*g + 0.114*b
        cb = -0.168736*r - 0.331264*g + 0.5*b + 0.5
        cr =  0.5*r - 0.418688*g - 0.081312*b + 0.5
        return torch.cat([y, cb, cr], 1) * 255.0 - 128.0


class YCbCrToRGB(nn.Module):
    def forward(self, x):
        x = (x + 128.0) / 255.0
        y, cb, cr = x[:, 0:1], x[:, 1:2] - 0.5, x[:, 2:3] - 0.5
        r = y + 1.402 * cr
        g = y - 0.344136 * cb - 0.714136 * cr
        b = y + 1.772 * cb
        return torch.cat([r, g, b], 1)


class DiffJPEG(nn.Module):
    def __init__(self, height=256, width=256, differentiable=True, quality=80):
        super().__init__()
        self.height = height
        self.width = width
        self.default_quality = quality
        self.to_ycbcr = RGBToYCbCr()
        self.to_rgb = YCbCrToRGB()
        self.dct = DCT()
        self.idct = IDCT()

    def forward(self, x, quality=None):
        if quality is None:
            quality = self.default_quality
        device = x.device
        B, C, H, W = x.shape

        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        pH, pW = x.shape[2], x.shape[3]

        ycbcr = self.to_ycbcr(x)
        coeffs = self.dct(ycbcr)

        if isinstance(quality, torch.Tensor):
            quality = quality.item()
        quality = max(1, min(100, int(quality)))
        yt = _quant_table(quality, 'y').to(device).view(1, 1, 1, 8, 8)
        ct = _quant_table(quality, 'c').to(device).view(1, 1, 1, 8, 8)

        y_q = _DiffRound.apply(coeffs[:, 0:1] / yt) * yt
        c_q = _DiffRound.apply(coeffs[:, 1:3] / ct) * ct
        dequant = torch.cat([y_q, c_q], 1)

        rgb = self.to_rgb(self.idct(dequant, pH, pW))
        if pad_h > 0 or pad_w > 0:
            rgb = rgb[:, :, :H, :W]
        return torch.clamp(rgb, 0.0, 1.0)
