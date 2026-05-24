"""
utils/noise.py
Multiplicative gamma speckle-noise model from Majee et al. 2020:

    J = I * η,   η ~ Gamma(L, 1/L)

Mean(η) = 1, Var(η) = 1/L  →  smaller L = heavier speckle.
All tensors and arrays use the [0, 255] dynamic range.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image


def add_gamma_noise(
    u: torch.Tensor,
    L: int,
    seed: int = None,
) -> torch.Tensor:
    """
    Apply multiplicative gamma speckle noise to a clean-image tensor.

    Parameters
    ----------
    u    : clean image tensor, shape (B, 1, H, W) or (1, H, W), range [0, 255]
    L    : number of looks.  L=1 → heavy noise;  L=10 → mild noise.
    seed : optional RNG seed for reproducibility

    Returns
    -------
    noisy tensor, same shape as u, clamped to [0, 255]
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # η ~ Gamma(shape=L, scale=1/L)  →  E[η]=1, Var[η]=1/L
    eta = rng.gamma(shape=L, scale=1.0 / L, size=u.shape).astype(np.float32)
    eta_t = torch.from_numpy(eta).to(u.device)
    return (u * eta_t).clamp(0.0, 255.0)


def generate_noisy_dataset(
    clean_dir: str,
    noisy_dir: str,
    L: int,
    seed_base: int = 0,
) -> None:
    """
    Pre-generate noisy versions of every image in clean_dir and save to noisy_dir.
    One noisy image per clean image, using a deterministic per-image seed.

    Parameters
    ----------
    clean_dir : directory of clean grayscale PNG images
    noisy_dir : output directory for noisy images
    L         : gamma-noise parameter
    seed_base : base seed; image i gets seed = seed_base + i
    """
    os.makedirs(noisy_dir, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in Path(clean_dir).iterdir()
                   if p.suffix.lower() in exts)

    if not paths:
        print(f"  [noise] No images found in {clean_dir}")
        return

    print(f"  Generating {len(paths)} noisy images (L={L}) → {noisy_dir}")
    try:
        from tqdm import tqdm
        path_iter = tqdm(paths, ncols=70)
    except ImportError:
        path_iter = paths
    for i, p in enumerate(path_iter):
        img = np.array(Image.open(str(p)).convert("L"), dtype=np.float32)
        u_t = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        f_t = add_gamma_noise(u_t, L=L, seed=seed_base + i)
        f_np = f_t.squeeze().numpy().astype(np.uint8)
        out_path = os.path.join(noisy_dir, p.name)
        Image.fromarray(f_np).save(out_path)
