"""
ablate_inertia.py — Study importance of the u_tt (inertial) term.

Three variants:
  1. Full model     — u_tt + γ·u_t = div(g · φ(K*u) · K^T)   (current)
  2. No inertia     — u_t = div(g · φ(K*u) · K^T)            (γ→∞, or drop u_prv)
                      Equivalent: u^{n+1} = u^n + τ² · div_term(u^n)
  3. No damping     — u_tt = div(g · φ(K*u) · K^T)           (γ=0)

Compares PSNR, SSIM, and visual output on test images.
Analyzes per-stage convergence and stability.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NUM_STAGES, NUM_FILTERS, FILTER_SIZE,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS,
    TAU, CHECKPOINT_DIR, CLEAN_TEST_DIR,
)
from models import InertialTNRDNetwork
from dataset import make_test_loader
from utils.metrics import psnr, ssim

ABL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "inertia_ablation")
os.makedirs(ABL_DIR, exist_ok=True)


def _find_best_ckpt(ckpt_dir, L):
    """Find the best checkpoint and determine its true number of stages."""
    import re
    max_stage = 0
    best_path = None
    # 1. Check individual stage checkpoints — find the highest stage
    for fname in os.listdir(ckpt_dir):
        m = re.match(rf"stage(\d+)_L{L}_best\.pth", fname)
        if m:
            s = int(m.group(1))
            if s > max_stage:
                max_stage = s
                best_path = os.path.join(ckpt_dir, fname)
    if best_path and max_stage > 0:
        sd = torch.load(best_path, map_location="cpu")
        stage_nums = sorted(set(int(k.split('.')[1]) for k in sd if k.startswith('stages.')))
        # Some stage checkpoints contain all stages (greedy saves full model)
        # Use the actual number of stages in the state dict
        return best_path, len(stage_nums)
    # 2. Fallback to final model
    final_path = os.path.join(ckpt_dir, f"model_L{L}_final.pth")
    if os.path.exists(final_path):
        sd = torch.load(final_path, map_location="cpu")
        stage_nums = sorted(set(int(k.split('.')[1]) for k in sd if k.startswith('stages.')))
        return final_path, len(stage_nums)
    return None, NUM_STAGES


def load_model(L, ckpt_dir, num_stages=None, device=DEVICE):
    path, actual_stages = _find_best_ckpt(ckpt_dir, L)
    if num_stages is None:
        num_stages = actual_stages
    model = InertialTNRDNetwork(
        num_stages=num_stages, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True,
        device=device,
    ).to(device)
    if path and os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        print(f"  Loaded: {path}  (stages={num_stages})")
    else:
        print(f"  WARNING: no checkpoint for L={L}, using untrained (stages={num_stages})")
    return model


def forward_with_variant(model, f, variant, active_stages=None):
    """
    Run the model with a modified forward to ablate the inertial term.
    
    Variants:
      'full'      : original inertial PDE (u_tt + γ·u_t)
      'no_inertia': first-order diffusion only (u_t, no u_tt)
                    u^{n+1} = u^n + τ² · div_term(u^n)
      'no_damping': undamped inertial (u_tt, γ=0)
                    u^{n+1} = 2·u^n - u^{n-1} + τ² · div_term(u^n)
    """
    T = active_stages if active_stages is not None else len(model.stages)
    u_prv = f.clone()
    u_cur = f.clone()
    stage_outputs = []

    for t in range(T):
        stage = model.stages[t]
        tau = stage.tau
        gam = stage.gamma_inertia

        # Compute g and div_term (same as original)
        if stage.use_g_func:
            from models.stage import _gray_level_indicator
            g = _gray_level_indicator(
                u_cur, stage.nu, stage.K_thresh,
                stage.blur_kernel.to(dtype=u_cur.dtype),
            )
        else:
            g = torch.ones_like(u_cur)
        div_term = stage._divergence_term(u_cur, g)

        if variant == "full":
            # (1+γτ) u^{n+1} = (2+γτ) u^n - u^{n-1} + τ²·div
            numer = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term
            u_nxt = numer / (1.0 + gam * tau)
        elif variant == "no_inertia":
            # First-order: forward Euler, no momentum
            # u^{n+1} = u^n + τ²·div
            u_nxt = u_cur + tau ** 2 * div_term
        elif variant == "no_damping":
            # Undamped: γ=0
            # u^{n+1} = 2·u^n - u^{n-1} + τ²·div
            u_nxt = 2.0 * u_cur - u_prv + tau ** 2 * div_term
        else:
            raise ValueError(f"Unknown variant: {variant}")

        u_nxt = u_nxt.clamp(0.0, 255.0)
        stage_outputs.append(u_nxt)
        u_prv = u_cur
        u_cur = u_nxt

    return u_cur, stage_outputs


@torch.no_grad()
def evaluate_variant(model, loader, device, variant, active_stages=None):
    model.eval()
    psnr_list, ssim_list = [], []
    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f = f.to(device)
        u_pred, _ = forward_with_variant(model, f, variant,
                                          active_stages=active_stages)
        u_pred = u_pred.clamp(0.0, 255.0)
        psnr_list.append(psnr(u_pred, u_gt))
        ssim_list.append(ssim(u_pred, u_gt))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def run_ablation(L=1, ckpt_dir=CHECKPOINT_DIR, test_dir=CLEAN_TEST_DIR,
                 num_stages=NUM_STAGES, device=DEVICE):
    print(f"\n{'='*70}")
    print(f"  INERTIA (u_tt) ABLATION STUDY  L={L}")
    print(f"{'='*70}")

    model = load_model(L, ckpt_dir, num_stages=None, device)
    model.eval()

    variants = ["full", "no_inertia", "no_damping"]
    labels = {
        "full": "Full (u_tt + γ·u_t)",
        "no_inertia": "No inertia (u_t only)",
        "no_damping": "Undamped (u_tt, γ=0)",
    }
    colors = {"full": "steelblue", "no_inertia": "firebrick", "no_damping": "forestgreen"}
    markers = {"full": "o", "no_inertia": "s", "no_damping": "^"}

    # ── 1. Full test set evaluation ───────────────────────────────────────────
    print("\n  ── Full test set evaluation ──")
    test_loader = make_test_loader(test_dir, L=L, seed=0)
    metrics = {}
    for v in variants:
        p, s = evaluate_variant(model, test_loader, device, v)
        metrics[v] = {"psnr": p, "ssim": s}
        print(f"  {labels[v]:30s}  PSNR={p:.4f}  SSIM={s:.4f}")

    # ── 2. Per-stage progression ─────────────────────────────────────────────
    print("\n  ── Per-stage progression ──")
    T_max = len(model.stages)
    stage_metrics = {v: {"psnr": [], "ssim": []} for v in variants}
    for T in range(1, T_max + 1):
        for v in variants:
            p, s = evaluate_variant(model, test_loader, device, v,
                                     active_stages=T)
            stage_metrics[v]["psnr"].append(p)
            stage_metrics[v]["ssim"].append(s)
        print(f"  T={T:2d}  "
              f"Full={stage_metrics['full']['psnr'][-1]:.2f}  "
              f"NoInertia={stage_metrics['no_inertia']['psnr'][-1]:.2f}  "
              f"NoDamping={stage_metrics['no_damping']['psnr'][-1]:.2f}")

    # ── 3. Theoretical analysis: eigenvalue stability ─────────────────────────
    print("\n  ── Theoretical analysis ──")
    print(f"  τ = {TAU},  γ = {GAMMA_INERTIA}")
    print(f"  Spectral radius (full):  |λ| = 1 ± τ√(stuff)  (conditionally stable)")
    print(f"  Spectral radius (no_inertia): forward Euler, |λ| ≈ 1 + τ²·κ")
    print(f"  Spectral radius (no_damping): undamped oscillator, oscillatory")

    # ── 4. Save plots ─────────────────────────────────────────────────────────
    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, metric in enumerate(["psnr", "ssim"]):
        ax = axes[idx]
        vals = [metrics[v][metric] for v in variants]
        bars = ax.bar(range(len(variants)), vals, color=[colors[v] for v in variants],
                      alpha=0.85, tick_label=[labels[v] for v in variants])
        ax.bar_label(bars, fmt=f"%.4f", fontsize=9)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Inertia Ablation — {metric.upper()}  (L={L})")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(ABL_DIR, f"inertia_ablation_bar_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")

    # Per-stage progression
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, metric in enumerate(["psnr", "ssim"]):
        ax = axes[idx]
        for v in variants:
            ax.plot(range(1, T_max + 1), stage_metrics[v][metric],
                    color=colors[v], marker=markers[v], linewidth=1.5,
                    label=labels[v])
        ax.set_xlabel("Stages (T)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Inertia Ablation — {metric.upper()} vs Stages  (L={L})")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(ABL_DIR, f"inertia_ablation_vs_stages_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 5. Visual comparison on first test image ──────────────────────────────
    print("\n  ── Visual comparison on test_000 ──")
    for u_gt, f in make_test_loader(test_dir, L=L, seed=0):
        u_gt = u_gt.to(device)
        f = f.to(device)
        results = {}
        for v in variants:
            u_pred, stages = forward_with_variant(model, f, v)
            results[v] = {
                "pred": u_pred,
                "psnr": psnr(u_pred.clamp(0, 255), u_gt),
                "ssim": ssim(u_pred.clamp(0, 255), u_gt),
                "stages": stages,
            }
            print(f"  {labels[v]:30s}  PSNR={results[v]['psnr']:.4f}  "
                  f"SSIM={results[v]['ssim']:.4f}")
        break  # just first image

    # Visual comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    row_titles = ["Full (u_tt + γ·u_t)", "No inertia (u_t only)", "Undamped (u_tt, γ=0)"]
    for idx, v in enumerate(variants):
        _to_np = lambda t: t.detach().cpu().float().squeeze().numpy().clip(0, 255)
        axes[0, idx].imshow(_to_np(results[v]["pred"]), cmap="gray", vmin=0, vmax=255)
        axes[0, idx].set_title(f"{labels[v]}\nPSNR={results[v]['psnr']:.2f}")
        axes[0, idx].axis("off")
        # Difference from full model
        diff = (results[v]["pred"] - results["full"]["pred"]).abs()
        axes[1, idx].imshow(_to_np(diff), cmap="hot", vmin=0, vmax=30)
        axes[1, idx].set_title(f"|Diff from Full|\nmean={diff.mean():.2f}")
        axes[1, idx].axis("off")
    fig.suptitle(f"Inertia Ablation — Visual Comparison  (L={L})", fontsize=13)
    fig.tight_layout()
    path = os.path.join(ABL_DIR, f"inertia_visual_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 6. Save metrics ──────────────────────────────────────────────────────
    json_path = os.path.join(ABL_DIR, f"inertia_ablation_L{L}.json")
    all_data = {
        "L": L,
        "metrics": {v: {k: float(v2) if isinstance(v2, torch.Tensor) else v2
                        for k, v2 in metrics[v].items()} for v in variants},
        "stage_metrics": stage_metrics,
        "analysis": {
            "tau": TAU,
            "gamma": GAMMA_INERTIA,
            "note_full": "Inertial PDE: (1+γτ)u^{n+1} = (2+γτ)u^n - u^{n-1} + τ²·div",
            "note_no_inertia": "First-order: u^{n+1} = u^n + τ²·div (forward Euler)",
            "note_no_damping": "Undamped: u^{n+1} = 2·u^n - u^{n-1} + τ²·div (γ=0)",
        },
    }
    # Convert numpy arrays to lists
    for v in variants:
        all_data["stage_metrics"][v]["psnr"] = [float(x) for x in stage_metrics[v]["psnr"]]
        all_data["stage_metrics"][v]["ssim"] = [float(x) for x in stage_metrics[v]["ssim"]]
    with open(json_path, "w") as fp:
        json.dump(all_data, fp, indent=2)
    print(f"  Metrics → {json_path}")

    return metrics, stage_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--ckpt_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--test_dir", default=CLEAN_TEST_DIR)
    args = parser.parse_args()
    run_ablation(L=args.L, ckpt_dir=args.ckpt_dir, test_dir=args.test_dir)


if __name__ == "__main__":
    main()
