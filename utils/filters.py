"""
utils/filters.py
Fixed (non-learned) filter bank  k_i  for Project 8.

Implements DCT-based zero-mean filters following the TNRD paper
(Chen & Pock, CVPR 2016, §3.1).  A 7×7 filter bank with 48 filters
(= 7²-1, excluding the DC component) is the standard TNRD configuration.

Filters are returned as torch tensors with requires_grad=False so the
optimiser never touches them.  The 180° rotation utility is provided for
the transpose-convolution step in the divergence computation.
"""

import math
import torch
import numpy as np


def build_dct_filters(
    num_filters: int = 48,
    filter_size: int = 7,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build a bank of DCT-II basis filters, zero-meaned and L2-normalised.

    The first num_filters non-DC DCT-II basis vectors (lexicographic order,
    excluding the all-ones DC component) are reshaped into (filter_size × filter_size)
    kernels, zero-meaned, and L2-normalised.

    Parameters
    ----------
    num_filters : number of filters N_k (default 48 = 7²-1 for 7×7 kernels)
    filter_size : spatial size m of each m×m kernel (must be odd, default 7)
    device      : target device

    Returns
    -------
    Tensor of shape (num_filters, 1, filter_size, filter_size),
    requires_grad=False, on `device`.
    """
    assert filter_size % 2 == 1, "filter_size must be odd"
    m   = filter_size
    N   = m * m  # total pixels per filter

    # Build all DCT-II 2D basis functions (vectorised)
    idx  = np.arange(m, dtype=np.float64)
    # 1-D DCT-II basis: cos(π/m * (k + 0.5) * n) for k,n = 0..m-1
    basis1d = np.cos(np.pi / m * np.outer(np.arange(m), idx + 0.5))  # (m, m)

    # 2-D outer products to get m² basis images of shape (m, m)
    all_filters = []
    for ky in range(m):
        for kx in range(m):
            f2d = np.outer(basis1d[:, ky], basis1d[:, kx])  # (m, m)
            all_filters.append(f2d)

    # all_filters[0] is the DC component — skip it
    all_filters = all_filters[1:]  # (m²-1) entries

    # Select first num_filters entries
    selected = all_filters[:num_filters]

    bank = []
    for f in selected:
        f = f - f.mean()                 # zero-mean
        norm = np.linalg.norm(f)
        if norm > 1e-8:
            f = f / norm                 # L2-normalise
        bank.append(f.astype(np.float32))

    # Stack → (num_filters, 1, m, m)
    bank_t = torch.tensor(np.stack(bank, axis=0)).unsqueeze(1)
    bank_t = bank_t.to(device)
    bank_t.requires_grad_(False)
    return bank_t


def get_rotated_filters(
    filters: torch.Tensor,
) -> torch.Tensor:
    """
    Return the 180°-rotated version of the filter bank (i.e. flip both spatial axes).
    Used as the transposed filter in the divergence term.

    Parameters
    ----------
    filters : (N, 1, m, m)

    Returns
    -------
    Tensor of same shape, not requiring grad.
    """
    rotated = filters.flip(-1).flip(-2)
    return rotated.detach()
