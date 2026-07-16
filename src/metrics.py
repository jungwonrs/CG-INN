#CG-INN
import torch
import torch.nn.functional as F
from math import exp


def gaussian(size, sigma):
    g = torch.Tensor([exp(-(x - size//2)**2 / (2*sigma**2)) for x in range(size)])
    return g / g.sum()


def create_window(size, ch):
    w1d = gaussian(size, 1.5).unsqueeze(1)
    w2d = w1d.mm(w1d.t()).unsqueeze(0).unsqueeze(0)
    return w2d.expand(ch, 1, size, size).contiguous()


def calc_ssim(img1, img2, win_size=11):
    ch = img1.size(1)
    win = create_window(win_size, ch).type_as(img1)
    pad = win_size // 2

    mu1 = F.conv2d(img1, win, padding=pad, groups=ch)
    mu2 = F.conv2d(img2, win, padding=pad, groups=ch)
    mu1_sq, mu2_sq = mu1**2, mu2**2

    sig1_sq = F.conv2d(img1*img1, win, padding=pad, groups=ch) - mu1_sq
    sig2_sq = F.conv2d(img2*img2, win, padding=pad, groups=ch) - mu2_sq
    sig12 = F.conv2d(img1*img2, win, padding=pad, groups=ch) - mu1*mu2

    C1, C2 = 0.01**2, 0.03**2
    ssim = ((2*mu1*mu2 + C1) * (2*sig12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2))
    return ssim.mean().item()


def calc_psnr(img1, img2):
    mse = torch.mean((img1 - img2)**2)
    if mse == 0:
        return 100.0
    return (10 * torch.log10(1.0 / mse)).item()


def calc_nc(img1, img2):
    v1 = img1.reshape(img1.shape[0], -1)
    v2 = img2.reshape(img2.shape[0], -1)
    return F.cosine_similarity(v1, v2).mean().item()
