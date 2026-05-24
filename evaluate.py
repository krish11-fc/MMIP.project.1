"""
evaluate.py  —  Full test evaluation for Inertial TNRD Despeckling
==================================================================
Reproduces comparison from Majee 2020 Table 1 for L=1 and L=10.

Metrics (paper §5.1):
  PSNR  — Peak Signal-to-Noise Ratio  (higher = better)
  MSSIM — Mean Structural Similarity Index  (higher = better)
  SI    — Speckle Index = std(I)/mean(I)  (lower = better, for SAR)

The proposed (learned) model is compared against the underlying PDE model
(paper's own analytic formulation) at L=1 and L=10 as required.

BUG 3 FIX:
    model.load(ckpt_path, map_location=device) passed a torch.device object
    as map_location, which is truthy and bypassed the "or 'cpu'" fallback.
    network.py's load() now always maps to "cpu" first (fixed there), but we
    also call load_state_dict directly here for belt-and-braces safety.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NOISE_LEVELS, CLEAN_TEST_DIR,
    CHECKPOINT_DIR, RESULTS_DIR, TABLES_DIR, PLOT_DIR,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS, NUM_STAGES, TAU,
)
from models import InertialTNRDNetwork
from dataset import make_test_loader
from utils.metrics import psnr, ssim
from utils.visualization import (
    plot_denoising_result, plot_stage_outputs, save_results_table
)
from utils.noise import add_gamma_noise


# ──────────────────────────────────────────────────────────────────────────────
# PDE baseline (paper's own analytic model, no learned RBF)
# ──────────────────────────────────────────────────────────────────────────────

def run_pde_baseline(
    f:          torch.Tensor,   # (1,1,H,W) noisy input in [0,255]
    gamma:      float = GAMMA_INERTIA,
    tau:        float = TAU,
    nu:         float = NU,
    K_thresh:   float = K,      # 128.0 for [0,255] images (BUG 1 fix propagated)
    sigma:      float = SIGMA_SMOOTH,
    max_iter:   int   = 200,
    eps:        float = 1e-4,
) -> torch.Tensor:
    """
    Run the analytic telegraph diffusion PDE (paper eq. 2.5) without any
    learned components.

    Update rule (paper §4, eq. d):
        (1+γτ)·I^{n+1} = (2+γτ)·I^n - I^{n-1} + τ²·div(g·∇I^n)

    g uses the correct K_thresh (128.0) for [0,255] images.
    """
    import torch.nn.functional as F

    device = f.device
    dtype  = f.dtype

    # Gaussian kernel for u_σ
    ksz  = 2 * int(3 * sigma) + 1
    half = ksz // 2
    t1d  = torch.arange(-half, half+1, dtype=dtype, device=device)
    g1d  = torch.exp(-t1d**2 / (2*sigma**2))
    g1d  = g1d / g1d.sum()
    blur_k = g1d.outer(g1d).view(1, 1, ksz, ksz)

    def gaussian_blur(u):
        return F.conv2d(u, blur_k, padding=half)

    def gray_level_indicator(u):
        u_xi = gaussian_blur(u)
        u_abs = u_xi.abs().clamp(min=1e-8)
        B = u.shape[0]
        M = u_abs.view(B, -1).max(dim=1).values.view(B, 1, 1, 1).clamp(min=1e-8)
        b = (2.0 * u_abs**nu) / (M**nu + u_abs**nu)
        # Edge-stopping on u_xi gradient
        # K_thresh = 128.0 for [0,255] images (BUG 1 fix)
        u_pad_x = F.pad(u_xi, (1,1,0,0), mode="replicate")
        u_pad_y = F.pad(u_xi, (0,0,1,1), mode="replicate")
        gx = (u_pad_x[:,:,:,2:] - u_pad_x[:,:,:,:-2]) / 2.0
        gy = (u_pad_y[:,:,2:,:] - u_pad_y[:,:,:-2,:]) / 2.0
        mag = (gx**2 + gy**2).sqrt()
        c = 1.0 / (1.0 + (mag / K_thresh)**2)
        return b * c

    def divergence(u, g_map):
        u_px = F.pad(u, (1,1,0,0), mode="replicate")
        u_py = F.pad(u, (0,0,1,1), mode="replicate")
        gx = (u_px[:,:,:,2:] - u_px[:,:,:,:-2]) / 2.0
        gy = (u_py[:,:,2:,:] - u_py[:,:,:-2,:]) / 2.0
        flux_x = g_map * gx
        flux_y = g_map * gy
        fx_p = F.pad(flux_x, (1,1,0,0), mode="replicate")
        fy_p = F.pad(flux_y, (0,0,1,1), mode="replicate")
        div_x = (fx_p[:,:,:,2:] - fx_p[:,:,:,:-2]) / 2.0
        div_y = (fy_p[:,:,2:,:] - fy_p[:,:,:-2,:]) / 2.0
        return div_x + div_y

    I_prv = f.clone()
    I_cur = f.clone()

    with torch.no_grad():
        for _ in range(max_iter):
            g_map = gray_level_indicator(I_cur)
            div_t = divergence(I_cur, g_map)
            numer = (2.0 + gamma*tau)*I_cur - I_prv + tau**2 * div_t
            I_nxt = (numer / (1.0 + gamma*tau)).clamp(0.0, 255.0)

            rel_err = ((I_nxt - I_cur).norm()**2 /
                       I_cur.norm().clamp(min=1e-8)**2).item()
            I_prv = I_cur
            I_cur = I_nxt
            if rel_err < eps:
                break

    return I_cur


# ──────────────────────────────────────────────────────────────────────────────
# Speckle Index
# ──────────────────────────────────────────────────────────────────────────────

def speckle_index(img: torch.Tensor) -> float:
    """SI = std(I)/mean(I)  — paper §5.1."""
    arr = img.detach().cpu().float().numpy().flatten()
    m = arr.mean()
    if m < 1e-8:
        return float("nan")
    return float(arr.std() / m)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate one model on one test image
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_image(
    model_or_fn,
    u_gt:   torch.Tensor,
    f:      torch.Tensor,
    device: torch.device,
    name:   str,
    L:      int,
    save_dir: str,
    use_stages: bool = True,
) -> dict:
    """Evaluate on a single (clean, noisy) pair. Returns metric dict."""
    u_gt = u_gt.to(device)
    f    = f.to(device)

    if callable(model_or_fn):
        out = model_or_fn(f)
        if isinstance(out, tuple):
            u_pred, stage_outputs = out
        else:
            u_pred = out
            stage_outputs = []
    else:
        u_pred = model_or_fn
        stage_outputs = []

    u_pred = u_pred.clamp(0.0, 255.0)

    psnr_noisy = psnr(f,      u_gt)
    ssim_noisy = ssim(f,      u_gt)
    psnr_pred  = psnr(u_pred, u_gt)
    ssim_pred  = ssim(u_pred, u_gt)
    si_noisy   = speckle_index(f)
    si_pred    = speckle_index(u_pred)

    os.makedirs(save_dir, exist_ok=True)
    plot_denoising_result(
        u_gt, f, u_pred,
        psnr_noisy=psnr_noisy, psnr_pred=psnr_pred,
        ssim_noisy=ssim_noisy, ssim_pred=ssim_pred,
        save_path=os.path.join(save_dir, f"{name}_L{L}.png"),
        title=f"{name}  L={L}",
    )
    if stage_outputs and use_stages:
        plot_stage_outputs(
            stage_outputs, u_gt, f,
            save_path=os.path.join(save_dir, f"{name}_L{L}_stages.png"),
        )

    return {
        "psnr_noisy": psnr_noisy,
        "ssim_noisy": ssim_noisy,
        "psnr_pred":  psnr_pred,
        "ssim_pred":  ssim_pred,
        "si_noisy":   si_noisy,
        "si_pred":    si_pred,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    ckpt_dir:   str  = CHECKPOINT_DIR,
    test_dir:   str  = CLEAN_TEST_DIR,
    out_dir:    str  = RESULTS_DIR,
    tables_dir: str  = TABLES_DIR,
    device:     torch.device = DEVICE,
    noise_levels: list = NOISE_LEVELS,
):
    """
    Evaluate the learned model vs the underlying PDE baseline at L=1 and L=10.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    all_results = {}
    print(f"  Test images: {test_dir}")

    for L in noise_levels:
        print(f"\n{'='*60}")
        print(f"  Evaluating  L={L}")
        print(f"{'='*60}")

        # Load learned model
        model = InertialTNRDNetwork(
            num_stages    = NUM_STAGES,
            num_filters   = NUM_FILTERS,
            filter_size   = FILTER_SIZE,
            gamma_inertia = GAMMA_INERTIA,
            sigma_smooth  = SIGMA_SMOOTH,
            nu            = NU,
            K_thresh      = K,
            num_centers   = RBF_NUM_CENTERS,
            use_g_func    = True,
            device        = device,
        ).to(device)

        ckpt_path = os.path.join(ckpt_dir, f"model_L{L}_final.pth")
        if os.path.exists(ckpt_path):
            # BUG 3 FIX: load to CPU first, then the model is already on device
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"  Loaded checkpoint: {ckpt_path}")
        else:
            print(f"  WARNING: no checkpoint at {ckpt_path} — using untrained model")
        model.eval()

        test_loader = make_test_loader(test_dir, L=L, seed=0)
        results_L = {"learned": {}, "pde_baseline": {}}
        table_rows = []

        for u_gt, f in test_loader:
            img_idx = len(table_rows)
            img_name = f"test_{img_idx:03d}"

            # ── Learned model ──────────────────────────────────────────────
            res_learned = evaluate_image(
                model, u_gt, f, device,
                name=f"learned_{img_name}",
                L=L, save_dir=out_dir,
            )

            # ── PDE baseline (analytic, no learned params) ─────────────────
            f_dev = f.to(device)
            u_pde = run_pde_baseline(f_dev, gamma=GAMMA_INERTIA, tau=TAU,
                                     nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH)
            u_gt_dev = u_gt.to(device)
            res_pde = {
                "psnr_noisy": psnr(f_dev,  u_gt_dev),
                "ssim_noisy": ssim(f_dev,  u_gt_dev),
                "psnr_pred":  psnr(u_pde,  u_gt_dev),
                "ssim_pred":  ssim(u_pde,  u_gt_dev),
                "si_noisy":   speckle_index(f_dev),
                "si_pred":    speckle_index(u_pde),
            }

            results_L["learned"][img_name]      = res_learned
            results_L["pde_baseline"][img_name] = res_pde

            table_rows.append({
                "image":         img_name,
                "L":             L,
                "psnr_noisy":    res_learned["psnr_noisy"],
                "psnr_learned":  res_learned["psnr_pred"],
                "ssim_learned":  res_learned["ssim_pred"],
                "psnr_pde":      res_pde["psnr_pred"],
                "ssim_pde":      res_pde["ssim_pred"],
            })

            print(f"  {img_name}  L={L}  "
                  f"Noisy PSNR={res_learned['psnr_noisy']:.2f}  "
                  f"Learned PSNR={res_learned['psnr_pred']:.2f}  "
                  f"PDE PSNR={res_pde['psnr_pred']:.2f}")

        all_results[f"L{L}"] = results_L
        _write_comparison_table(table_rows, L, tables_dir)

    json_path = os.path.join(out_dir, "all_results.json")
    with open(json_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n  All results saved → {json_path}")

    return all_results


def _write_comparison_table(rows: list, L: int, tables_dir: str):
    """Write a plain-text table matching paper Table 1 format."""
    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, f"table_L{L}.txt")
    header = (
        f"\nComparison Table  —  L={L}\n"
        f"{'Image':<18}  {'Noisy PSNR':>10}  "
        f"{'PDE PSNR':>10}  {'PDE SSIM':>9}  "
        f"{'Learned PSNR':>13}  {'Learned SSIM':>13}\n"
        + "-"*80
    )
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['image']:<18}  {r['psnr_noisy']:>10.4f}  "
            f"{r['psnr_pde']:>10.4f}  {r['ssim_pde']:>9.4f}  "
            f"{r['psnr_learned']:>13.4f}  {r['ssim_learned']:>13.4f}"
        )
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"  Table saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Inertial TNRD model")
    parser.add_argument("--ckpt_dir",  default=CHECKPOINT_DIR)
    parser.add_argument("--test_dir",  default=CLEAN_TEST_DIR)
    parser.add_argument("--out_dir",   default=RESULTS_DIR)
    parser.add_argument("--tables_dir",default=TABLES_DIR)
    parser.add_argument("--L",         type=int, nargs="+", default=NOISE_LEVELS,
                        help="Noise levels to evaluate (default: 1 10)")
    args = parser.parse_args()

    evaluate_all(
        ckpt_dir    = args.ckpt_dir,
        test_dir    = args.test_dir,
        out_dir     = args.out_dir,
        tables_dir  = args.tables_dir,
        device      = DEVICE,
        noise_levels= args.L,
    )


if __name__ == "__main__":
    main()
