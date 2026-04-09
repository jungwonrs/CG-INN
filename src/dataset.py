import os
import glob
import random
import hashlib
import time
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from contracts.key_derivation import KeyPipeline


class ICINNDataset(Dataset):
    def __init__(self, data_root, image_size=256, mode='train',
                 limit=None, use_vc_keys=True):
        super().__init__()
        self.files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            self.files.extend(glob.glob(os.path.join(data_root, ext)))
        if limit and len(self.files) > limit:
            random.shuffle(self.files)
            self.files = self.files[:limit]
        if not self.files:
            raise FileNotFoundError(f"No images in {data_root}")
        self.use_vc_keys = use_vc_keys
        self.pipeline = KeyPipeline() if use_vc_keys else None
        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        si = random.randint(0, len(self.files) - 1)
        ci = random.randint(0, len(self.files) - 1)
        try:
            di = self.tf(Image.open(self.files[si]).convert("RGB"))
            cover = self.tf(Image.open(self.files[ci]).convert("RGB"))
            if self.use_vc_keys and self.pipeline:
                uid = f"cc_{hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]}"
                if uid not in self.pipeline.users:
                    self.pipeline.register(uid)
                di_hash = hashlib.sha256(
                    (self.files[si] + str(time.time())).encode()).hexdigest()
                key = self.pipeline.gen_key(uid, di_hash)
            else:
                key = hashlib.sha256(
                    (str(time.time()) + str(random.random())).encode()).hexdigest()
            return di, cover, key
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))


class MixedDataset(Dataset):
    def __init__(self, data_roots, image_size=256, mode='train',
                 limit_per_dataset=2000, use_vc_keys=True):
        super().__init__()
        self.files = []
        for root in data_roots:
            pool = []
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                pool.extend(glob.glob(os.path.join(root, ext)))
            if limit_per_dataset and len(pool) > limit_per_dataset:
                random.shuffle(pool)
                pool = pool[:limit_per_dataset]
            self.files.extend(pool)
        random.shuffle(self.files)
        if not self.files:
            raise FileNotFoundError(f"No images in {data_roots}")
        self.use_vc_keys = use_vc_keys
        self.pipeline = KeyPipeline() if use_vc_keys else None
        if mode == 'train':
            self.tf = transforms.Compose([
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        si = random.randint(0, len(self.files) - 1)
        ci = random.randint(0, len(self.files) - 1)
        try:
            di = self.tf(Image.open(self.files[si]).convert("RGB"))
            cover = self.tf(Image.open(self.files[ci]).convert("RGB"))
            if self.use_vc_keys and self.pipeline:
                uid = f"cc_{hashlib.sha256(str(random.random()).encode()).hexdigest()[:8]}"
                if uid not in self.pipeline.users:
                    self.pipeline.register(uid)
                di_hash = hashlib.sha256(
                    (self.files[si] + str(time.time())).encode()).hexdigest()
                key = self.pipeline.gen_key(uid, di_hash)
            else:
                key = hashlib.sha256(
                    (str(time.time()) + str(random.random())).encode()).hexdigest()
            return di, cover, key
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))
