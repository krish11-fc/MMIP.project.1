"""
ablation.py  —  Ablation Study for Inertial TNRD Despeckling
=============================================================
Evaluates contribution of each component at L=1 and L=10.

Ablation variants:
  A1. Full model          — g_func=True,  trained RBF φ_i,  T=5 stages
  A2. No gray-level g     — g_func=False  (g≡1, pure RBF diffusion)
  A3. No learned RBF      — Perona-Malik warm-start only (not trained)
  A4. Varying T stages    — T ∈ {1, 2, 3, 4, 5} using full trained model
  A5. PDE baseline        — paper's analytic model, no learned components

Outputs saved to outputs/ablation/:
  ablation_results.json        — full numeric results
  ablation_summary.txt         — human-readable comparison table
  ablation_psnr_mean_L{L}.png  — bar chart PSNR per variant
  ablation_ssim_mean_L{L}.png  — bar chart SSIM per variant
  ablation_stages_L{L}.png     — PSNR/SSIM vs T line plot
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NOISE_LEVELS, CLEAN_TEST_DIR,
    CHECKPOINT_DIR, ABLATION_DIR,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS, NUM_STAGES, TAU,
)
from models import InertialTNRDNetwork
from dataset import make_test_loader
from utils.metrics import psnr, ssim
from evaluate import run_pde_baseline, speckle_index


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_model_on_test(
    model_or_fn,
    test_loader,
    device: torch.device,
    active_stages: int = None,
) -> dict:
    """
    Returns mean ± std of PSNR, SSIM, SI over all test images.
    model_or_fn can be an nn.Module or a plain callable(f) -> (u_pred, stages).
    """
    psnr_list, ssim_list, si_list = [], [], []
    is_nn = isinstance(model_or_fn, torch.nn.Module)

    for u_gt, f in test_loader:
        u_gt = u_gt.to(device)
        f    = f.to(device)

        if is_nn:
            model_or_fn.eval()
            u_pred, _ = model_or_fn(f, active_stages=active_stages)
        else:
            result = model_or_fn(f)
            u_pred = result[0] if isinstance(result, tuple) else result

        u_pred = u_pred.clamp(0.0, 255.0)
        psnr_list.append(psnr(u_pred, u_gt))
        ssim_list.append(ssim(u_pred, u_gt))
        si_list.append(speckle_index(u_pred))

    return {
        "psnr_mean": float(np.mean(psnr_list)),
        "psnr_std":  float(np.std(psnr_list)),
        "ssim_mean": float(np.mean(ssim_list)),
        "ssim_std":  float(np.std(ssim_list)),
        "si_mean":   float(np.mean(si_list)),
        "si_std":    float(np.std(si_list)),
        "psnr_list": psnr_list,
        "ssim_list": ssim_list,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_model(use_g_func: bool, device: torch.device,
                 num_stages: int = NUM_STAGES) -> InertialTNRDNetwork:
    return InertialTNRDNetwork(
        num_stages    = num_stages,
        num_filters   = NUM_FILTERS,
        filter_size   = FILTER_SIZE,
        gamma_inertia = GAMMA_INERTIA,
        sigma_smooth  = SIGMA_SMOOTH,
        nu            = NU,
        K_thresh      = K,
        num_centers   = RBF_NUM_CENTERS,
        use_g_func    = use_g_func,
        device        = device,
    ).to(device)


def _load_ckpt(model, ckpt_dir, L, g_suffix=""):
    """Load best checkpoint if available; return model unchanged if not found."""
    # Try best first, then final, then last
    for name in [f"model_L{L}{g_suffix}_final.pth",
                 f"stage{NUM_STAGES}_L{L}{g_suffix}_best.pth",
                 f"stage{NUM_STAGES}_L{L}{g_suffix}_last.pth"]:
        path = os.path.join(ckpt_dir, name)
        if os.path.exists(path):
            model.load(path)
            print(f"    Loaded: {path}")
            return model
    print(f"    WARNING: no checkpoint found in {ckpt_dir} for L={L}{g_suffix}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Main ablation runner
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(
    test_dir: str  = CLEAN_TEST_DIR,
    ckpt_dir: str  = CHECKPOINT_DIR,
    out_dir:  str  = ABLATION_DIR,
    device:   torch.device = DEVICE,
    noise_levels: list = None,
) -> dict:
    if noise_levels is None:
        noise_levels = NOISE_LEVELS   # [1, 10]

    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for L in noise_levels:
        print(f"\n{'='*60}")
        print(f"  ABLATION  L={L}")
        print(f"{'='*60}")

        test_loader = make_test_loader(test_dir, L=L, seed=0)
        results_L   = {}

        # ── A1. Full model (g + trained RBF) ─────────────────────────────────
        print("  A1: Full model (g=True, trained RBF) ...")
        m1 = _build_model(use_g_func=True, device=device)
        m1 = _load_ckpt(m1, ckpt_dir, L, g_suffix="")
        results_L["A1_full_model"] = eval_model_on_test(m1, test_loader, device)
        print(f"      PSNR={results_L['A1_full_model']['psnr_mean']:.4f}  "
              f"SSIM={results_L['A1_full_model']['ssim_mean']:.4f}")

        # ── A2. No gray-level indicator (g≡1) ────────────────────────────────
        print("  A2: No g_func (g≡1, trained RBF only) ...")
        m2 = _build_model(use_g_func=False, device=device)
        m2 = _load_ckpt(m2, ckpt_dir, L, g_suffix="_nog")
        results_L["A2_no_g_func"] = eval_model_on_test(m2, test_loader, device)
        print(f"      PSNR={results_L['A2_no_g_func']['psnr_mean']:.4f}  "
              f"SSIM={results_L['A2_no_g_func']['ssim_mean']:.4f}")

        # ── A3. Fixed RBF (Perona-Malik warm-start, NOT trained) ──────────────
        print("  A3: Fixed RBF (Perona-Malik init, no training) ...")
        m3 = _build_model(use_g_func=True, device=device)
        # Deliberately do NOT load any checkpoint — keeps warm-start weights
        results_L["A3_fixed_rbf"] = eval_model_on_test(m3, test_loader, device)
        print(f"      PSNR={results_L['A3_fixed_rbf']['psnr_mean']:.4f}  "
              f"SSIM={results_L['A3_fixed_rbf']['ssim_mean']:.4f}")

        # ── A4. Varying T (using full trained model with active_stages=T) ─────
        print("  A4: Varying T stages ...")
        m4 = _build_model(use_g_func=True, device=device)
        m4 = _load_ckpt(m4, ckpt_dir, L, g_suffix="")
        results_L["A4_varying_T"] = {}
        for T in range(1, NUM_STAGES + 1):
            r = eval_model_on_test(m4, test_loader, device, active_stages=T)
            results_L["A4_varying_T"][f"T{T}"] = r
            print(f"      T={T}  PSNR={r['psnr_mean']:.4f}  SSIM={r['ssim_mean']:.4f}")

        # ── A5. PDE baseline (paper's analytic model, no learned params) ──────
        print("  A5: PDE baseline (analytic, no learned components) ...")
        def pde_fn(f):
            u = run_pde_baseline(f, gamma=GAMMA_INERTIA, tau=TAU,
                                 nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH)
            return u, []
        results_L["A5_pde_baseline"] = eval_model_on_test(pde_fn, test_loader, device)
        print(f"      PSNR={results_L['A5_pde_baseline']['psnr_mean']:.4f}  "
              f"SSIM={results_L['A5_pde_baseline']['ssim_mean']:.4f}")

        all_results[f"L{L}"] = results_L

        # ── Save plots for this L ─────────────────────────────────────────────
        _plot_bar(results_L, L, out_dir)
        _plot_stages(results_L["A4_varying_T"], L, out_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "ablation_results.json")
    with open(json_path, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n  JSON  → {json_path}")

    # ── Save summary table ────────────────────────────────────────────────────
    _write_summary(all_results, out_dir)

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _plot_bar(results_L: dict, L: int, out_dir: str) -> None:
    """Bar chart: A1–A3 + A5 side by side for PSNR and SSIM."""
    variants = {
        "A1 Full\n(g+RBF)":         results_L["A1_full_model"],
        "A2 No g\n(RBF only)":      results_L["A2_no_g_func"],
        "A3 Fixed RBF\n(PM init)":  results_L["A3_fixed_rbf"],
        "A5 PDE\n(analytic)":       results_L["A5_pde_baseline"],
    }
    colors = ["steelblue", "darkorange", "forestgreen", "firebrick"]

    for metric, ylabel in [("psnr_mean", "PSNR (dB)"), ("ssim_mean", "SSIM")]:
        labels = list(variants.keys())
        values = [variants[k][metric] for k in labels]
        errs   = [variants[k][metric.replace("mean", "std")] for k in labels]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.bar(labels, values, color=colors, alpha=0.88,
                      yerr=errs, capsize=5, error_kw={"elinewidth": 1.5})
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Ablation Study — {ylabel}  (L={L})", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(bottom=max(0, min(values) - 3))
        fig.tight_layout()
        path = os.path.join(out_dir, f"ablation_{metric}_L{L}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"    Saved → {path}")


def _plot_stages(varying_T: dict, L: int, out_dir: str) -> None:
    """Line plot: PSNR and SSIM vs number of unrolled stages T."""
    T_vals    = sorted(int(k[1:]) for k in varying_T)
    psnr_vals = [varying_T[f"T{t}"]["psnr_mean"] for t in T_vals]
    ssim_vals = [varying_T[f"T{t}"]["ssim_mean"] for t in T_vals]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(T_vals, psnr_vals, "o-", color="steelblue", linewidth=2, markersize=7)
    axes[0].set_xlabel("Stages T", fontsize=11)
    axes[0].set_ylabel("PSNR (dB)", fontsize=11)
    axes[0].set_title(f"PSNR vs #Stages  (L={L})", fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(T_vals)
    for x, y in zip(T_vals, psnr_vals):
        axes[0].annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)

    axes[1].plot(T_vals, ssim_vals, "s-", color="darkorange", linewidth=2, markersize=7)
    axes[1].set_xlabel("Stages T", fontsize=11)
    axes[1].set_ylabel("SSIM", fontsize=11)
    axes[1].set_title(f"SSIM vs #Stages  (L={L})", fontsize=12)
    axes[1].grid(alpha=0.3)
    axes[1].set_xticks(T_vals)
    for x, y in zip(T_vals, ssim_vals):
        axes[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, f"ablation_stages_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"    Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Human-readable summary
# ──────────────────────────────────────────────────────────────────────────────

def _write_summary(all_results: dict, out_dir: str) -> None:
    """Write outputs/ablation/ablation_summary.txt — human-readable table."""
    W = 74
    lines = [
        "=" * W,
        "  ABLATION STUDY SUMMARY",
        "  Architecture: u_tt + γ·u_t = div( g(u_σ) · φ_i(K_i*u) · K_i^T )",
        "  Fixed : k_i (DCT filter bank), γ=0.5",
        "  Learned: φ_i^t (RBF per stage), λ^t (fidelity per stage)",
        f"  K={K}  (scale-corrected for [0,255] images)",
        "=" * W,
    ]

    variant_keys = {
        "A1_full_model":  "A1  Full model (g=True, trained RBF φ_i)",
        "A2_no_g_func":   "A2  No gray-level g  (g≡1, only RBF)",
        "A3_fixed_rbf":   "A3  Fixed RBF (Perona-Malik, no training)",
        "A5_pde_baseline":"A5  PDE Baseline (paper analytic model)",
    }

    for L_key, results_L in sorted(all_results.items()):
        lines.append(f"\n  Noise level: {L_key}")
        lines.append(
            f"  {'Variant':<44}  {'PSNR (dB)':>10}  "
            f"{'SSIM':>8}  {'SI':>8}"
        )
        lines.append("  " + "-" * (W - 2))

        for vkey, vlabel in variant_keys.items():
            if vkey in results_L:
                r = results_L[vkey]
                lines.append(
                    f"  {vlabel:<44}  "
                    f"{r['psnr_mean']:>10.4f}  "
                    f"{r['ssim_mean']:>8.4f}  "
                    f"{r['si_mean']:>8.4f}"
                )

        if "A4_varying_T" in results_L:
            lines.append(f"\n  A4  Effect of number of stages T:")
            lines.append(f"  {'T':>5}  {'PSNR (dB)':>10}  {'SSIM':>8}")
            lines.append("  " + "-" * 28)
            for T in sorted(int(k[1:]) for k in results_L["A4_varying_T"]):
                r = results_L["A4_varying_T"][f"T{T}"]
                lines.append(
                    f"  {T:>5}  {r['psnr_mean']:>10.4f}  {r['ssim_mean']:>8.4f}"
                )

    lines += [
        "\n" + "=" * W,
        "  Key findings:",
        "  A1 vs A2 : contribution of gray-level indicator g(u_σ)",
        "  A1 vs A3 : benefit of training RBF φ_i vs fixed Perona-Malik",
        "  A1 vs A5 : benefit of learned components vs analytic PDE",
        "  A4       : PSNR/SSIM improvement per additional unrolled stage",
        "=" * W,
    ]

    path = os.path.join(out_dir, "ablation_summary.txt")
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"\n  Summary → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ablation study — Inertial TNRD")
    parser.add_argument("--test_dir", default=CLEAN_TEST_DIR)
    parser.add_argument("--ckpt_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--out_dir",  default=ABLATION_DIR)
    parser.add_argument("--L",        type=int, nargs="+",
                        default=NOISE_LEVELS, help="Noise levels (default: 1 10)")
    args = parser.parse_args()

    run_ablation(
        test_dir     = args.test_dir,
        ckpt_dir     = args.ckpt_dir,
        out_dir      = args.out_dir,
        device       = DEVICE,
        noise_levels = args.L,
    )


if __name__ == "__main__":
    main()
