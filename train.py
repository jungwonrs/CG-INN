import os
import random
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import kornia.augmentation as K

from config import Config
from src.dataset import ICINNDataset, MixedDataset
from src.loss import ICINNLoss
from src.metrics import calc_psnr
from src.models import (INN_Model, SecretEncoder, SecretDecoder,
                        KeyManager, ZPredictor, NoiseEstimator)
from src.diffjpeg import DiffJPEG


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _clamp01_(x):
    return torch.clamp(x, 0.0, 1.0)


def quantization_simulation(x):
    return x + (torch.rand_like(x) - 0.5) / 255.0


def hash_to_vector(keys, device):
    vecs = []
    for k in keys:
        sha = hashlib.sha256(k.encode()).digest()
        v = (np.frombuffer(sha, dtype=np.uint8).astype(np.float32) / 127.5) - 1.0
        vecs.append(np.concatenate([v, v]))
    return torch.tensor(np.array(vecs)).to(device)


def save_checkpoint(epoch, inn, km, enc, dec, z_pred, noise_est, optimizer, path):
    torch.save({
        'epoch': epoch, 'inn': inn.state_dict(), 'km': km.state_dict(),
        'enc': enc.state_dict(), 'dec': dec.state_dict(),
        'z_pred': z_pred.state_dict(), 'noise_est': noise_est.state_dict(),
        'optimizer': optimizer.state_dict()
    }, path)


def pretrain_autoencoder(cfg, encoder, decoder):
    epochs = getattr(cfg, 'PRETRAIN_EPOCHS', 100)
    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    crit = nn.MSELoss()
    use_vc = getattr(cfg, 'USE_VC_KEYS', True)
    train_roots = getattr(cfg, 'TRAIN_DATA_ROOTS', None)
    if train_roots:
        ds = MixedDataset(train_roots, cfg.IMAGE_SIZE, mode='train',
                          limit_per_dataset=getattr(cfg, 'TRAIN_LIMIT_PER', 2000),
                          use_vc_keys=use_vc)
    else:
        ds = ICINNDataset(cfg.TRAIN_DATA_ROOT, cfg.IMAGE_SIZE,
                          limit=cfg.TRAIN_LIMIT, use_vc_keys=use_vc)
    ld = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                    num_workers=cfg.NUM_WORKERS, drop_last=True)
    encoder.train(); decoder.train()
    for ep in range(epochs):
        for di, _, _ in tqdm(ld, desc=f"Pretrain {ep+1}/{epochs}", leave=True):
            di = di.to(cfg.DEVICE)
            loss = crit(decoder(encoder(di), head='clean'), di)
            opt.zero_grad(); loss.backward(); opt.step()


def train_ic_inn(external_cfg=None, logger_callback=None):
    cfg = external_cfg or Config()
    set_seed(42)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    km = KeyManager(key_dim=cfg.KEY_DIM).to(cfg.DEVICE)
    inn = INN_Model(num_blocks=cfg.INN_BLOCKS, key_dim=cfg.KEY_DIM).to(cfg.DEVICE)
    enc = SecretEncoder().to(cfg.DEVICE)
    dec = SecretDecoder(noise_dim=32).to(cfg.DEVICE)
    z_pred = ZPredictor().to(cfg.DEVICE)
    noise_est = NoiseEstimator(noise_dim=32).to(cfg.DEVICE)

    total_params = sum(p.numel() for m in [inn, enc, dec, z_pred, km, noise_est] for p in m.parameters())
    print(f"Total params: {total_params:,} ({total_params/1e6:.1f}M)")

    pretrain_autoencoder(cfg, enc, dec)

    optimizer = optim.Adam([
        {'params': inn.parameters(), 'lr': cfg.LR},
        {'params': enc.parameters(), 'lr': cfg.LR},
        {'params': dec.parameters(), 'lr': cfg.LR * 2},
        {'params': z_pred.parameters(), 'lr': cfg.LR * 3},
        {'params': km.parameters(), 'lr': cfg.LR * 2},
        {'params': noise_est.parameters(), 'lr': cfg.LR * 3},
    ], weight_decay=1e-5, foreach=False)
    all_params = [p for g in optimizer.param_groups for p in g['params']]

    S1_END = cfg.STAGE1_END
    S2_END = cfg.STAGE2_END
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [180, 260], gamma=0.5)
    criterion = ICINNLoss(cfg).to(cfg.DEVICE)
    diff_jpeg = DiffJPEG(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, differentiable=True).to(cfg.DEVICE)

    use_vc = getattr(cfg, 'USE_VC_KEYS', True)
    train_roots = getattr(cfg, 'TRAIN_DATA_ROOTS', None)
    if train_roots:
        ds = MixedDataset(train_roots, cfg.IMAGE_SIZE, mode='train',
                          limit_per_dataset=getattr(cfg, 'TRAIN_LIMIT_PER', 2000),
                          use_vc_keys=use_vc)
    else:
        ds = ICINNDataset(cfg.TRAIN_DATA_ROOT, cfg.IMAGE_SIZE,
                          limit=cfg.TRAIN_LIMIT, use_vc_keys=use_vc)
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                        num_workers=cfg.NUM_WORKERS, drop_last=True)

    z_weight = cfg.LAMBDA_Z
    hf_weight = cfg.LAMBDA_HF
    base_r = cfg.LAMBDA_R
    accum = cfg.ACCUM_STEPS

    for epoch in range(1, cfg.EPOCHS + 1):
        inn.train(); km.train(); enc.train(); dec.train()
        z_pred.train(); noise_est.train()
        optimizer.zero_grad()

        if epoch <= S1_END:
            stage, real_r = 1, 0.0
        elif epoch <= S2_END:
            stage = 2
            real_r = base_r * ((epoch - S1_END) / (S2_END - S1_END)) * 0.5
        else:
            stage, real_r = 3, base_r

        loop = tqdm(loader, desc=f"Ep{epoch} S{stage}", leave=True)
        for i, (di, ci, ck_batch) in enumerate(loop):
            di, ci = di.to(cfg.DEVICE), ci.to(cfg.DEVICE)
            vk = km(hash_to_vector(ck_batch, cfg.DEVICE))
            pdi = enc(di)
            z_out, si_raw = inn(pdi, ci, vk)

            if torch.isnan(si_raw).any() or torch.isinf(si_raw).any():
                optimizer.zero_grad(); continue

            si = quantization_simulation(_clamp01_(si_raw))
            with torch.no_grad():
                h_psnr = calc_psnr(ci, si)

            if stage == 1:
                atk_si = si
            elif stage == 2:
                r = random.random()
                if r < 0.80:
                    atk_si = diff_jpeg(si, quality=random.uniform(85, 95))
                elif r < 0.92:
                    sig = random.uniform(0.3, 0.7)
                    atk_si = _clamp01_(K.RandomGaussianBlur((5,5),(sig,sig),p=1.0).to(cfg.DEVICE)(si))
                else:
                    atk_si = _clamp01_(K.RandomGaussianNoise(0.0,0.01,p=1.0).to(cfg.DEVICE)(si))
            else:
                r = random.random()
                if r < 0.75:
                    atk_si = diff_jpeg(si, quality=random.uniform(75, 95))
                elif r < 0.88:
                    sig = random.uniform(0.1, 0.6)
                    atk_si = _clamp01_(K.RandomGaussianBlur((5,5),(sig,sig),p=1.0).to(cfg.DEVICE)(si))
                else:
                    nse = random.uniform(0.01, 0.05)
                    atk_si = _clamp01_(K.RandomGaussianNoise(0.0,nse,p=1.0).to(cfg.DEVICE)(si))

            if torch.isnan(atk_si).any() or torch.isinf(atk_si).any():
                optimizer.zero_grad(); continue

            z_clean = z_pred(si)
            z_atk = z_pred(atk_si)
            noise_vec = noise_est(atk_si)

            pdi_hat = inn(si, None, vk, rev=True, z_guess=z_clean)
            di_hat_clean = dec(pdi_hat, head='clean')

            if stage == 1:
                di_hat_robust = di_hat_clean
                di_hat_wrong = di_hat_clean
            else:
                pdi_hat_atk = inn(atk_si, None, vk, rev=True, z_guess=z_atk)
                di_hat_robust = dec(pdi_hat_atk, head='robust', noise_vec=noise_vec)
                wrong_vk = torch.randn_like(vk)
                pdi_hat_wrong = inn(atk_si, None, wrong_vk, rev=True, z_guess=z_atk)
                di_hat_wrong = dec(pdi_hat_wrong, head='robust', noise_vec=noise_vec)

            loss, l_h, l_c, l_r, l_s, l_hf = criterion(
                ci, si, di, di_hat_clean, di_hat_robust, di_hat_wrong,
                lambda_r_scale=real_r, lambda_hf=hf_weight)

            loss_zp = (nn.functional.l1_loss(z_clean, z_out.detach())
                       + nn.functional.l1_loss(z_atk, z_out.detach())) * z_weight
            loss_zr = torch.mean(torch.abs(z_out)) * (z_weight * 0.1)

            total = (loss + loss_zp + loss_zr) / accum
            total.backward()

            if (i + 1) % accum == 0:
                has_nan = any(p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()) for p in all_params)
                if has_nan:
                    optimizer.zero_grad(); continue
                nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step(); optimizer.zero_grad()

            loop.set_postfix(stg=stage, h=f"{h_psnr:.1f}", hide=f"{l_h.item():.4f}", rev=f"{l_c.item():.4f}")

        torch.cuda.empty_cache()
        scheduler.step()
        if logger_callback:
            logger_callback(epoch, inn, km, enc, dec, z_pred, noise_est)
        if epoch % 10 == 0 or epoch in [S1_END, S2_END, cfg.EPOCHS]:
            save_checkpoint(epoch, inn, km, enc, dec, z_pred, noise_est, optimizer,
                            os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch}.pth"))


if __name__ == "__main__":
    train_ic_inn()
