import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import numpy as np


class DWT(nn.Module):
    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        ll = x1 + x2 + x3 + x4
        hl = -x1 - x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hh = x1 - x2 - x3 + x4
        return torch.cat((ll, hl, lh, hh), 1)


class IWT(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        oc = c // 4
        oh, ow = h * 2, w * 2
        x1 = x[:, 0:oc] / 2
        x2 = x[:, oc:oc*2] / 2
        x3 = x[:, oc*2:oc*3] / 2
        x4 = x[:, oc*3:oc*4] / 2
        out = torch.zeros(b, oc, oh, ow, device=x.device, dtype=x.dtype)
        out[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        out[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        out[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        out[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return out


def _norm(ch):
    return nn.GroupNorm(min(8, ch), ch)


class CBAM(nn.Module):
    def __init__(self, ch, ratio=16):
        super().__init__()
        r = max(ch // ratio, 4)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(ch, r, 1, bias=False), nn.ReLU(), nn.Conv2d(r, ch, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.fc(self.avg(x)) + self.fc(self.mx(x)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], 1)))
        return x * sa


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), _norm(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, padding=1), _norm(ch), CBAM(ch))

    def forward(self, x):
        return x + self.body(x)


class KeyManager(nn.Module):
    def __init__(self, key_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128), nn.LeakyReLU(0.2, True),
            nn.Linear(128, key_dim), nn.Tanh())

    def hash_to_vector(self, keys):
        device = next(self.parameters()).device
        if isinstance(keys, torch.Tensor):
            return keys.to(device).float()
        vecs = []
        for k in keys:
            sha = hashlib.sha256(k.encode()).digest()
            v = (np.frombuffer(sha, dtype=np.uint8).astype(np.float32) / 127.5) - 1.0
            vecs.append(np.concatenate([v, v]))
        return torch.tensor(np.array(vecs), dtype=torch.float32).to(device)

    def forward(self, key_input):
        return self.net(self.hash_to_vector(key_input))


class AdaINBlock(nn.Module):
    def __init__(self, ch, key_dim):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.n1 = _norm(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.n2 = _norm(ch)
        self.act = nn.LeakyReLU(0.2, True)
        self.style = nn.Linear(key_dim, ch * 4)

    def forward(self, x, vk):
        s = self.style(vk).unsqueeze(2).unsqueeze(3)
        s1, b1, s2, b2 = s.chunk(4, dim=1)
        s1 = torch.tanh(s1) * 0.1 + 1.0
        s2 = torch.tanh(s2) * 0.1 + 1.0
        out = self.act(self.n1(self.c1(x)) * s1 + b1)
        return self.act(self.n2(self.c2(out)) * s2 + b2 + x)


class CouplingLayer(nn.Module):
    def __init__(self, channels, key_dim):
        super().__init__()
        half = channels // 2
        self.s_net = nn.Sequential(
            nn.Conv2d(half, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            AdaINBlock(64, key_dim),
            nn.Conv2d(64, half, 3, 1, 1), nn.Tanh())
        self.t_net = nn.Sequential(
            nn.Conv2d(half, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            AdaINBlock(64, key_dim),
            nn.Conv2d(64, half, 3, 1, 1))

    def forward(self, x, vk, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        s = self.s_net[3](self.s_net[2](self.s_net[1](self.s_net[0](x1)), vk))
        t = self.t_net[3](self.t_net[2](self.t_net[1](self.t_net[0](x1)), vk))
        if not reverse:
            return torch.cat((x1, x2 * torch.exp(s) + t), 1)
        else:
            return torch.cat((x1, (x2 - t) * torch.exp(-s)), 1)


class INN_Model(nn.Module):
    def __init__(self, num_blocks=16, channels=3, key_dim=256):
        super().__init__()
        self.dwt = DWT()
        self.iwt = IWT()
        total = 24
        self.blocks = nn.ModuleList([CouplingLayer(total, key_dim) for _ in range(num_blocks)])
        for i in range(num_blocks):
            perm = torch.randperm(total)
            self.register_buffer(f'perm_{i}', perm)
            self.register_buffer(f'inv_perm_{i}', torch.argsort(perm))

    def forward(self, pdi, ci, vk, rev=False, z_guess=None):
        if not rev:
            x = torch.cat((pdi, self.dwt(ci)), 1)
            for i, blk in enumerate(self.blocks):
                x = blk(x, vk)
                x = x[:, getattr(self, f'perm_{i}')]
            return x[:, :12], self.iwt(x[:, 12:])
        else:
            psi = self.dwt(pdi)
            if z_guess is None:
                z_guess = torch.zeros_like(psi)
            x = torch.cat((z_guess, psi), 1)
            for i, blk in reversed(list(enumerate(self.blocks))):
                x = x[:, getattr(self, f'inv_perm_{i}')]
                x = blk(x, vk, reverse=True)
            return x[:, :12]


class SecretEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), _norm(32), CBAM(32),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), _norm(64), CBAM(64),
            nn.Conv2d(64, 12, 3, 1, 1))

    def forward(self, di):
        return self.net(di)


class ZPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(0.2, True),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, 1, 1), nn.LeakyReLU(0.2, True),
            ResBlock(64), ResBlock(64), CBAM(64),
            nn.Conv2d(64, 12, 3, 1, 1))

    def forward(self, si):
        return self.net(si)


class NoiseEstimator(nn.Module):
    def __init__(self, noise_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(64, noise_dim), nn.LeakyReLU(0.2, True))

    def forward(self, si_tilde):
        return self.fc(self.conv(si_tilde).flatten(1))


class STN(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(ch, 32, 7), nn.MaxPool2d(2), nn.ReLU(),
            nn.Conv2d(32, 64, 5), nn.MaxPool2d(2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 6))
        self.fc[2].weight.data.zero_()
        self.fc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.fc(self.loc(x).flatten(1)).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class NoiseCondBlock(nn.Module):
    def __init__(self, ch, noise_dim=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), _norm(ch), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1), _norm(ch))
        self.mod = nn.Linear(noise_dim, ch * 2)

    def forward(self, x, nv):
        out = self.body(x)
        m = self.mod(nv).unsqueeze(2).unsqueeze(3)
        s, b = m.chunk(2, dim=1)
        s = torch.tanh(s) * 0.2 + 1.0
        return x + out * s + b


class SecretDecoder(nn.Module):
    def __init__(self, noise_dim=32):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(12, 64, 3, 1, 1), nn.ReLU(), _norm(64), CBAM(64))
        self.head_clean = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid())
        self.stn = STN(64)
        self.robust_blocks = nn.ModuleList([NoiseCondBlock(64, noise_dim) for _ in range(6)])
        self.robust_tail = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True))
        self.head_robust = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(), _norm(32),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, pdi_hat, head='clean', noise_vec=None):
        feat = self.trunk(pdi_hat)
        if head == 'clean':
            return self.head_clean(feat)
        feat = self.stn(feat)
        for blk in self.robust_blocks:
            feat = blk(feat, noise_vec)
        return self.head_robust(self.robust_tail(feat))
