

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from utils.noise import add_gamma_noise
from config import DEVICE, DATALOADER_NUM_WORKERS


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _load_gray_tensor(path: str) -> torch.Tensor:
    """Load a grayscale image as (1, H, W) float32 in [0, 255]."""
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return torch.from_numpy(img).unsqueeze(0)   # (1, H, W)


def _list_images(folder: str) -> list:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            f"Expected BSD400 (train) / BSD68 (val) under <project>/data/. "
            f"Download the benchmarks or set CLEAN_TRAIN_DIR / CLEAN_VAL_DIR in config.py."
        )
    return sorted(
        str(p) for p in root.iterdir()
        if p.suffix.lower() in _EXTS
    )


# ──────────────────────────────────────────────────────────────────────────────
# SpeckleDataset
# ──────────────────────────────────────────────────────────────────────────────

class SpeckleDataset(Dataset):
    
    def __init__(
        self,
        clean_dir:  str,
        L:          int,
        mode:       str   = "patch",
        patch_size: int   = 64,
        seed:       int   = None,
        augment:    bool  = False,
        max_images: int   = None,
    ):
        super().__init__()
        self.paths      = _list_images(clean_dir)
        if max_images is not None and max_images > 0:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise RuntimeError(f"No images found in {clean_dir}")
        self.L          = L
        self.mode       = mode
        self.patch_size = patch_size
        self.seed       = seed
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        u = _load_gray_tensor(self.paths[idx])   # (1, H, W) float32

        if self.mode == "patch":
            u = self._random_crop(u)
            if self.augment:
                u = self._random_flip(u)

        # Add gamma speckle noise  J = I · η,  η ~ Gamma(L, 1/L)
        seed_i = (self.seed + idx) if self.seed is not None else None
        # Add batch dim for noise function, then squeeze back
        u_b = u.unsqueeze(0)          # (1, 1, H, W)
        f_b = add_gamma_noise(u_b, L=self.L, seed=seed_i)
        f   = f_b.squeeze(0)          # (1, H, W)

        return u, f   # clean, noisy — both (1, H, W) in [0, 255]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _random_crop(self, u: torch.Tensor) -> torch.Tensor:
        """Random crop to (1, patch_size, patch_size)."""
        _, H, W = u.shape
        ps = self.patch_size
        if H < ps or W < ps:
            # pad if image smaller than patch
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            u = torch.nn.functional.pad(
                u.unsqueeze(0),
                (0, pad_w, 0, pad_h),
                mode="reflect"
            ).squeeze(0)
            _, H, W = u.shape
        top  = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        return u[:, top:top+ps, left:left+ps]

    def _random_flip(self, u: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            u = u.flip(-1)
        if random.random() > 0.5:
            u = u.flip(-2)
        return u


# ──────────────────────────────────────────────────────────────────────────────
# Pre-generated noisy dataset (faster I/O for large training runs)
# ──────────────────────────────────────────────────────────────────────────────

class PreGeneratedSpeckleDataset(Dataset):
    """
    Loads pre-generated (clean, noisy) image pairs from two directories.
    Faster than on-the-fly noise generation during training.
    Use utils.noise.generate_noisy_dataset() to pre-generate.

    Parameters
    ----------
    clean_dir  : directory of clean PNG images
    noisy_dir  : directory of corresponding noisy PNG images (same filenames)
    mode       : 'patch' or 'full'
    patch_size : crop size when mode='patch'
    augment    : random flips when mode='patch'
    """

    def __init__(
        self,
        clean_dir:  str,
        noisy_dir:  str,
        mode:       str  = "patch",
        patch_size: int  = 64,
        augment:    bool = False,
    ):
        super().__init__()
        self.clean_paths = _list_images(clean_dir)
        self.noisy_paths = _list_images(noisy_dir)
        assert len(self.clean_paths) == len(self.noisy_paths), \
            f"Mismatch: {len(self.clean_paths)} clean vs {len(self.noisy_paths)} noisy"
        self.mode       = mode
        self.patch_size = patch_size
        self.augment    = augment

    def __len__(self) -> int:
        return len(self.clean_paths)

    def __getitem__(self, idx: int):
        u = _load_gray_tensor(self.clean_paths[idx])
        f = _load_gray_tensor(self.noisy_paths[idx])

        if self.mode == "patch":
            u, f = self._sync_crop(u, f)
            if self.augment:
                u, f = self._sync_flip(u, f)

        return u, f

    def _sync_crop(self, u, f):
        _, H, W = u.shape
        ps = self.patch_size
        top  = random.randint(0, max(0, H - ps))
        left = random.randint(0, max(0, W - ps))
        return (u[:, top:top+ps, left:left+ps],
                f[:, top:top+ps, left:left+ps])

    def _sync_flip(self, u, f):
        if random.random() > 0.5:
            u, f = u.flip(-1), f.flip(-1)
        if random.random() > 0.5:
            u, f = u.flip(-2), f.flip(-2)
        return u, f


# ──────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_train_loader(clean_dir: str, L: int, batch_size: int = 4,
                      patch_size: int = 64, num_workers: int = None,
                      max_images: int = None):
    from torch.utils.data import DataLoader
    if num_workers is None:
        num_workers = DATALOADER_NUM_WORKERS
    ds = SpeckleDataset(
        clean_dir, L=L, mode="patch",
        patch_size=patch_size, augment=True, max_images=max_images,
    )
    pin = DEVICE.type == "cuda"
    kw = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


def make_val_loader(clean_dir: str, L: int, batch_size: int = 1,
                    patch_size: int = 64, seed: int = 42,
                    num_workers: int = None, max_images: int = None):
    from torch.utils.data import DataLoader
    if num_workers is None:
        num_workers = DATALOADER_NUM_WORKERS
    ds = SpeckleDataset(
        clean_dir, L=L, mode="patch",
        patch_size=patch_size, seed=seed, augment=False, max_images=max_images,
    )
    pin = DEVICE.type == "cuda"
    kw = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


def make_test_loader(clean_dir: str, L: int, seed: int = 0):
    """Full-image test loader (batch_size=1, no cropping)."""
    from torch.utils.data import DataLoader
    ds = SpeckleDataset(clean_dir, L=L, mode="full", seed=seed, augment=False)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)


class MixedSpeckleDataset(Dataset):
    """Each image gets a random noise level L drawn from noise_levels."""

    def __init__(self, clean_dir, noise_levels=None, mode="patch",
                 patch_size=64, augment=False, max_images=None):
        super().__init__()
        self.paths = _list_images(clean_dir)
        if max_images is not None and max_images > 0:
            self.paths = self.paths[:max_images]
        from config import NOISE_LEVELS_ALL
        self.noise_levels = noise_levels or NOISE_LEVELS_ALL
        self.mode = mode
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        u = _load_gray_tensor(self.paths[idx])
        if self.mode == "patch":
            u = self._random_crop(u)
            if self.augment:
                u = self._random_flip(u)
        L = random.choice(self.noise_levels)
        u_b = u.unsqueeze(0)
        f_b = add_gamma_noise(u_b, L=L, seed=None)
        f = f_b.squeeze(0)
        return u, f, L

    def _random_crop(self, u):
        _, H, W = u.shape
        ps = self.patch_size
        if H < ps or W < ps:
            pad_h, pad_w = max(0, ps-H), max(0, ps-W)
            u = torch.nn.functional.pad(u.unsqueeze(0), (0,pad_w,0,pad_h), mode="reflect").squeeze(0)
            _, H, W = u.shape
        top = random.randint(0, H-ps)
        left = random.randint(0, W-ps)
        return u[:, top:top+ps, left:left+ps]

    def _random_flip(self, u):
        if random.random() > 0.5:
            u = u.flip(-1)
        if random.random() > 0.5:
            u = u.flip(-2)
        return u


def make_mixed_train_loader(clean_dir, batch_size=8, patch_size=64,
                            num_workers=None, max_images=None):
    from torch.utils.data import DataLoader
    if num_workers is None:
        num_workers = DATALOADER_NUM_WORKERS
    ds = MixedSpeckleDataset(clean_dir, mode="patch", patch_size=patch_size,
                             augment=True, max_images=max_images)
    pin = DEVICE.type == "cuda"
    kw = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
              pin_memory=pin, drop_last=True)
    if num_workers > 0:
        kw["persistent_workers"] = True
        kw["prefetch_factor"] = 2
    return DataLoader(ds, **kw)


def make_mixed_val_loader(clean_dir, batch_size=1, patch_size=64, seed=42,
                          num_workers=None, max_images=None):
    """Validation at a single L; NCTDN validates at multiple Ls."""
    from torch.utils.data import DataLoader
    return make_val_loader(clean_dir, L=1, batch_size=batch_size,
                           patch_size=patch_size, seed=seed,
                           num_workers=num_workers, max_images=max_images)
