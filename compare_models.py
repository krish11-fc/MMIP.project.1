"""
compare_models.py — Comprehensive comparison of:
 
  A. Learned TNRD (current model, stage-wise trained RBF + g-function)
  B. PDE baseline (Majee 2020 analytic, no learned components)
  C. Learn-K (learnable filter bank k_i + learnable γ + RBF)
  D. TNRD-log (log-transformed, first-order diffusion, no g-function)
  E. NCTDN (noise-conditional telegraph diffusion, single model for all L)
  F. Full-Learn (learnable PDE scalars γ,τ,ν,K,σ + RBF, frozen filters)

Metrics: PSNR, SSIM, SI, per-image breakdown, wall-time.
Outputs: tables, JSON, bar charts.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NOISE_LEVELS, CLEAN_TEST_DIR,
    CHECKPOINT_DIR, NUM_STAGES, NUM_FILTERS, FILTER_SIZE,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, TAU,
)
from models import InertialTNRDNetwork, LearnKInertialTNRDNetwork, FullLearnInertialTNRDNetwork, TNRDLogNetwork
from models.noise_conditional_network import NoiseConditionalTNRDNetwork
from config import EMBED_DIM, NUM_NOISE_LEVELS
from dataset import make_test_loader
from utils.metrics import psnr, ssim
from evaluate import run_pde_baseline, speckle_index

COMPARE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "comparison")
os.makedirs(COMPARE_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_best_ckpt(ckpt_dir, L, suffix=""):
    import re
    max_stage = 0
    best_path = None
    for fname in os.listdir(ckpt_dir):
        m = re.match(rf"stage(\d+)_L{L}{suffix}_best\.pth", fname)
        if m:
            s = int(m.group(1))
            if s > max_stage:
                max_stage = s
                best_path = os.path.join(ckpt_dir, fname)
    if best_path and max_stage > 0:
        sd = torch.load(best_path, map_location="cpu")
        stage_nums = sorted(set(int(k.split('.')[1]) for k in sd if k.startswith('stages.')))
        return best_path, len(stage_nums)
    final_path = os.path.join(ckpt_dir, f"model_L{L}{suffix}_final.pth")
    if os.path.exists(final_path):
        sd = torch.load(final_path, map_location="cpu")
        stage_nums = sorted(set(int(k.split('.')[1]) for k in sd if k.startswith('stages.')))
        return final_path, len(stage_nums)
    return None, NUM_STAGES


def _load_ckpt(model, ckpt_dir, L, suffix=""):
    path, actual_stages = _find_best_ckpt(ckpt_dir, L, suffix)
    if path and os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        return model, True
    return model, False


def build_models(L, ckpt_dir, device=DEVICE):
    models = {}

    # A. Learned TNRD (current)
    _, ns_a = _find_best_ckpt(ckpt_dir, L, "")
    m_a = InertialTNRDNetwork(
        num_stages=ns_a, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)
    loaded, ok = _load_ckpt(m_a, ckpt_dir, L)
    if loaded:
        m_a.eval()
        models["A_Learned_TNRD"] = loaded
        print(f"  A: Learned TNRD  {'✓ loaded' if ok else '✗ no ckpt'}  (stages={ns_a})")

    # B. Majeed PDE baseline (analytic, no learned params)
    models["B_PDE_Baseline"] = None

    # C. Learn-K variant (learns filter bank k_i + gamma + RBF)
    _, ns_c = _find_best_ckpt(ckpt_dir, L, "_learnk")
    m_c = LearnKInertialTNRDNetwork(
        num_stages=ns_c, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_init=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)
    loaded_c, ok_c = _load_ckpt(m_c, ckpt_dir, L, "_learnk")
    if loaded_c:
        m_c.eval()
    models["C_Learn_K"] = loaded_c
    print(f"  C: Learn-K  {'✓ loaded' if ok_c else '✗ no ckpt (untrained)'}  (stages={ns_c})")

    # F. Full-Learn variant (learns all PDE scalars + RBF, frozen filters)
    _, ns_f = _find_best_ckpt(ckpt_dir, L, "_fulllearn")
    m_f = FullLearnInertialTNRDNetwork(
        num_stages=ns_f, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_init=GAMMA_INERTIA, nu_init=NU, K_init=K,
        num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)
    loaded_f, ok_f = _load_ckpt(m_f, ckpt_dir, L, "_fulllearn")
    if loaded_f:
        m_f.eval()
    models["F_Full_Learn"] = loaded_f
    print(f"  F: Full-Learn  {'✓ loaded' if ok_f else '✗ no ckpt (untrained)'}  (stages={ns_f})")

    # D. TNRD-log
    _, ns_d = _find_best_ckpt(ckpt_dir, L, "_tnrdlog")
    m_d = TNRDLogNetwork(
        num_stages=ns_d, num_filters=NUM_FILTERS,
        filter_size=FILTER_SIZE, num_centers=RBF_NUM_CENTERS, device=device,
    ).to(device)
    loaded_d, ok_d = _load_ckpt(m_d, ckpt_dir, L, "_tnrdlog")
    if loaded_d:
        m_d.eval()
    models["D_TNRD_Log"] = loaded_d
    print(f"  D: TNRD-Log  {'✓ loaded' if ok_d else '✗ no ckpt (untrained)'}  (stages={ns_d})")

    # E. NCTDN — single mixed-noise model for all L
    nctdn_path = os.path.join(ckpt_dir, "nctdn_model_mixed_final.pth")
    ok_e = False
    if os.path.exists(nctdn_path):
        sd = torch.load(nctdn_path, map_location="cpu")
        ns_e = max(int(k.split('.')[1]) for k in sd if k.startswith('stages.')) + 1 if any(k.startswith('stages.') for k in sd) else NUM_STAGES
        m_e = NoiseConditionalTNRDNetwork(
            num_stages=ns_e, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
            gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
            K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True,
            embed_dim=EMBED_DIM, num_noise_levels=NUM_NOISE_LEVELS, device=device,
        ).to(device)
        m_e.load_state_dict(sd)
        m_e.eval()
        models["E_NCTDN"] = lambda f, m=m_e, ll=L: m(f, L=ll)[0]
        ok_e = True
        print(f"  E: NCTDN  ✓ loaded  (stages={ns_e}, L passed at inference)")
    else:
        models["E_NCTDN"] = None
        print(f"  E: NCTDN  ✗ no ckpt ({nctdn_path})")

    return models


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_one(model_or_fn, u_gt, f, device, name=""):
    if model_or_fn is None:
        return {"psnr_noisy": 0.0, "ssim_noisy": 0.0, "psnr": 0.0, "ssim": 0.0, "si_noisy": 0.0, "si": 0.0}
    u_gt = u_gt.to(device)
    f = f.to(device)

    if name == "B_PDE_Baseline":
        u_pred = run_pde_baseline(f, gamma=GAMMA_INERTIA, tau=TAU,
                                   nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH)
    elif isinstance(model_or_fn, torch.nn.Module):
        model_or_fn.eval()
        u_pred, _ = model_or_fn(f)
    else:
        u_pred = model_or_fn(f)

    u_pred = u_pred.clamp(0.0, 255.0)

    return {
        "psnr_noisy": psnr(f, u_gt),
        "ssim_noisy": ssim(f, u_gt),
        "psnr": psnr(u_pred, u_gt),
        "ssim": ssim(u_pred, u_gt),
        "si_noisy": speckle_index(f),
        "si": speckle_index(u_pred),
    }


@torch.no_grad()
def eval_all_models(models, loader, device):
    results = {name: {"psnr": [], "ssim": [], "si": [], "per_image": {}}
               for name in models}
    for img_idx, (u_gt, f) in enumerate(loader):
        img_name = f"test_{img_idx:03d}"
        for name, model_or_fn in models.items():
            r = eval_one(model_or_fn, u_gt, f, device, name)
            results[name]["psnr"].append(r["psnr"])
            results[name]["ssim"].append(r["ssim"])
            results[name]["si"].append(r["si"])
            results[name]["per_image"][img_name] = r
        if img_idx % 5 == 0:
            print(f"    Image {img_idx}...")

    for name in models:
        results[name]["psnr_mean"] = float(np.mean(results[name]["psnr"]))
        results[name]["psnr_std"]  = float(np.std(results[name]["psnr"]))
        results[name]["ssim_mean"] = float(np.mean(results[name]["ssim"]))
        results[name]["ssim_std"]  = float(np.std(results[name]["ssim"]))
        results[name]["si_mean"]   = float(np.mean(results[name]["si"]))
        results[name]["si_std"]    = float(np.std(results[name]["si"]))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting & Reporting
# ──────────────────────────────────────────────────────────────────────────────

def _short_name(name):
    return (name.replace("A_", "A: ").replace("B_", "B: ").replace("C_", "C: ")
            .replace("D_", "D: ").replace("E_", "E: ").replace("F_", "F: ")
            .replace("_", " "))

def plot_comparison(all_results, L, save_dir):
    names = list(all_results.keys())
    short_names = [_short_name(n) for n in names]
    colors = ["steelblue", "firebrick", "forestgreen", "darkorange", "purple", "brown"]

    for metric, ylabel in [("psnr_mean", "PSNR (dB)"), ("ssim_mean", "SSIM")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        vals = [all_results[n][metric] for n in names]
        errs = [all_results[n][metric.replace("mean", "std")] for n in names]
        bars = ax.bar(short_names, vals, color=colors[:len(names)], alpha=0.85,
                      yerr=errs, capsize=5, error_kw={"elinewidth": 1.5})
        ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=4)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"Model Comparison — {ylabel}  (L={L})", fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        path = os.path.join(save_dir, f"comparison_{metric}_L{L}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"    Saved → {path}")

    # Per-image PSNR scatter
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(all_results[names[0]]["per_image"]))
    for idx, name in enumerate(names):
        img_psnrs = [all_results[name]["per_image"][f"test_{i:03d}"]["psnr"]
                     for i in range(len(all_results[name]["per_image"]))]
        ax.plot(x, img_psnrs, color=colors[idx], marker="o", linewidth=1.2,
                markersize=4, label=_short_name(name))
    ax.set_xlabel("Test image index")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Per-Image PSNR Comparison  (L={L})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, f"comparison_per_image_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"    Saved → {path}")


def write_report(all_results, L, save_dir):
    path = os.path.join(save_dir, f"comparison_report_L{L}.txt")
    lines = [
        "=" * 80,
        f"  MODEL COMPARISON REPORT  —  L={L}",
        "=" * 80,
        "",
        f"  {'Model':<30}  {'PSNR (dB)':>10}  {'SSIM':>8}  {'SI':>8}",
        "  " + "-" * 60,
    ]
    for name, res in all_results.items():
        lines.append(
            f"  {_short_name(name):<30}  "
            f"{res['psnr_mean']:>10.4f} ± {res['psnr_std']:.4f}  "
            f"{res['ssim_mean']:>8.4f} ± {res['ssim_std']:.4f}  "
            f"{res['si_mean']:>8.4f}"
        )
    lines += [
        "",
        "  Key observations:",
        "    A vs B: Benefit of learned RBF φ_i + g-function over analytic PDE",
        "    C vs A: Benefit of learning filter bank k_i + γ vs fixing them",
        "    D vs B: Benefit of TNRD on log-transformed speckle vs direct PDE",
        "    E vs A: Benefit of noise-conditional FiLM modulation across all L",
        "    F vs A: Benefit of learning all PDE scalars (γ,τ,ν,K,σ) vs fixing them",
        "",
        "  Notes:",
        "    - C (Learn-K), D (TNRD-Log), F (Full-Learn) need dedicated training.",
        "      Current results use untrained (warm-start) weights if no checkpoint.",
        "    - E (NCTDN) is a single model trained on all noise levels jointly.",
        "    - B (PDE Baseline) uses 200 iterations of the analytic PDE.",
        "    - SI = std(I)/mean(I); lower is better for SAR despeckling.",
        "=" * 80,
    ]
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"  Report → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_comparison(L=1, test_dir=None, ckpt_dir=CHECKPOINT_DIR, device=DEVICE):
    if test_dir is None:
        test_dir = CLEAN_TEST_DIR

    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON  L={L}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  Test images: {test_dir}")
    print(f"{'='*70}")

    print("\n  Building / loading models...")
    models = build_models(L, ckpt_dir, device)

    print("\n  Evaluating on test set...")
    test_loader = make_test_loader(test_dir, L=L, seed=0)
    results = eval_all_models(models, test_loader, device)

    print("\n  Generating plots...")
    plot_comparison(results, L, COMPARE_DIR)

    print("\n  Writing report...")
    write_report(results, L, COMPARE_DIR)

    json_path = os.path.join(COMPARE_DIR, f"comparison_L{L}.json")
    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"  JSON → {json_path}")

    # Summary print
    print(f"\n{'='*70}")
    print(f"  SUMMARY  L={L}")
    print(f"{'='*70}")
    for name, res in results.items():
        trained = "(trained)" if "ckpt" not in name else "(untrained)"
        print(f"  {_short_name(name):35s}  "
              f"PSNR={res['psnr_mean']:.4f} ± {res['psnr_std']:.4f}  "
              f"SSIM={res['ssim_mean']:.4f}")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, nargs="+", default=NOISE_LEVELS)
    parser.add_argument("--ckpt_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--test_dir", default=CLEAN_TEST_DIR)
    args = parser.parse_args()

    all_results = {}
    for L in args.L:
        r = run_comparison(L=L, test_dir=args.test_dir, ckpt_dir=args.ckpt_dir)
        all_results[f"L{L}"] = r

    json_path = os.path.join(COMPARE_DIR, "comparison_all.json")
    with open(json_path, "w") as fp:
        json.dump(all_results, fp, indent=2, default=str)
    print(f"\n  All results → {json_path}")


if __name__ == "__main__":
    main()
