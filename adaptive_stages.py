"""
adaptive_stages.py — Per-image adaptive stage selection.

Key finding from inertia ablation:
  T=5 → PSNR 16.72 dB (peak)
  T=10 → PSNR 11.69 dB (degraded)

This script evaluates ALL intermediate stages on each test image
and selects the best T per image (or globally).  Also compares
three strategies:

  1. Fixed T=N     : always use N stages
  2. Best per image: pick optimal T per image (oracle upper bound)
  3. Global best T : single T that maximizes mean PSNR over all images
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CLEAN_TEST_DIR, CHECKPOINT_DIR
from utils.metrics import psnr, ssim
from dataset import make_test_loader
from models import InertialTNRDNetwork
from ablate_inertia import load_model

ADAPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "adaptive_stages")
os.makedirs(ADAPT_DIR, exist_ok=True)


@torch.no_grad()
def evaluate_all_stages(model, loader, device, variant="full", max_stages=None):
    """
    Evaluate model at every intermediate T on every test image.
    Returns: {
        "per_image": {img_name: {T: {"psnr": ..., "ssim": ...}, ...}, ...},
        "best_per_image": {img_name: {"best_T": ..., "psnr": ..., "ssim": ...}},
        "global": {T: {"psnr_mean": ..., "ssim_mean": ...}}
    }
    """
    if max_stages is None:
        max_stages = len(model.stages)
    model.eval()

    # First pass: collect per-T outputs for each image
    per_image = {}
    for img_idx, (u_gt, f) in enumerate(loader):
        img_name = f"test_{img_idx:03d}"
        u_gt = u_gt.to(device)
        f = f.to(device)

        T = max_stages
        u_prv = f.clone()
        u_cur = f.clone()
        stage_outputs = []
        for t in range(T):
            stage = model.stages[t]
            tau = stage.tau
            gam = stage.gamma_inertia

            from models.stage import _gray_level_indicator
            if stage.use_g_func:
                g = _gray_level_indicator(u_cur, stage.nu, stage.K_thresh,
                                          stage.blur_kernel.to(dtype=u_cur.dtype))
            else:
                g = torch.ones_like(u_cur)
            div_term = stage._divergence_term(u_cur, g)

            if variant == "full":
                numer = (2.0 + gam * tau) * u_cur - u_prv + tau ** 2 * div_term
                u_nxt = numer / (1.0 + gam * tau)
            elif variant == "no_inertia":
                u_nxt = u_cur + tau ** 2 * div_term
            else:
                u_nxt = 2.0 * u_cur - u_prv + tau ** 2 * div_term

            u_nxt = u_nxt.clamp(0.0, 255.0)
            stage_outputs.append(u_nxt)
            u_prv = u_cur
            u_cur = u_nxt

        per_image[img_name] = {}
        for t in range(T):
            u_pred = stage_outputs[t]
            p = float(psnr(u_pred, u_gt).item()) if isinstance(psnr(u_pred, u_gt), torch.Tensor) else psnr(u_pred, u_gt)
            s = float(ssim(u_pred, u_gt).item()) if isinstance(ssim(u_pred, u_gt), torch.Tensor) else ssim(u_pred, u_gt)
            per_image[img_name][f"T={t+1}"] = {"psnr": p, "ssim": s}

    # Best T per image
    best_per_image = {}
    for img_name, metrics in per_image.items():
        best_t = max(range(1, max_stages + 1),
                     key=lambda t: metrics[f"T={t}"]["psnr"])
        best_per_image[img_name] = {
            "best_T": best_t,
            "psnr": metrics[f"T={best_t}"]["psnr"],
            "ssim": metrics[f"T={best_t}"]["ssim"],
        }

    # Global per-T stats
    global_stats = {}
    for t in range(1, max_stages + 1):
        psnrs = [per_image[img][f"T={t}"]["psnr"] for img in per_image]
        ssims = [per_image[img][f"T={t}"]["ssim"] for img in per_image]
        global_stats[f"T={t}"] = {
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)),
            "ssim_std": float(np.std(ssims)),
        }

    # Find global best T
    global_best_t = max(range(1, max_stages + 1),
                        key=lambda t: global_stats[f"T={t}"]["psnr_mean"])

    return {
        "per_image": per_image,
        "best_per_image": best_per_image,
        "global": global_stats,
        "global_best_T": global_best_t,
        "fixed_T10_psnr": global_stats[f"T={max_stages}"]["psnr_mean"],
        "global_best_psnr": global_stats[f"T={global_best_t}"]["psnr_mean"],
        "oracle_psnr": float(np.mean([b["psnr"] for b in best_per_image.values()])),
        "oracle_mean_T": float(np.mean([b["best_T"] for b in best_per_image.values()])),
    }


def run_adaptive_study(L=1, ckpt_dir=CHECKPOINT_DIR, test_dir=CLEAN_TEST_DIR,
                        device=DEVICE):
    print(f"\n{'='*70}")
    print(f"  ADAPTIVE STAGE SELECTION  L={L}")
    print(f"{'='*70}")

    model = load_model(L, ckpt_dir, device=device)
    model.eval()
    T_total = len(model.stages)
    print(f"  Model has {T_total} stages total")

    test_loader = make_test_loader(test_dir, L=L, seed=0)
    results = evaluate_all_stages(model, test_loader, device,
                                   variant="full", max_stages=T_total)

    print(f"\n  ── Per-stage global PSNR ──")
    for t in range(1, T_total + 1):
        s = results["global"][f"T={t}"]
        marker = " ← BEST" if t == results["global_best_T"] else ""
        print(f"  T={t:2d}  PSNR={s['psnr_mean']:.4f} ± {s['psnr_std']:.4f}  "
              f"SSIM={s['ssim_mean']:.4f}{marker}")

    print(f"\n  ── Selection strategies ──")
    print(f"  Fixed T={T_total}  : PSNR = {results['fixed_T10_psnr']:.4f}")
    print(f"  Global best T={results['global_best_T']}: PSNR = {results['global_best_psnr']:.4f}")
    print(f"  Oracle (per-image) : PSNR = {results['oracle_psnr']:.4f}  "
          f"(avg T={results['oracle_mean_T']:.1f})")

    print(f"\n  ── Per-image optimal T ──")
    for img_name, info in results["best_per_image"].items():
        print(f"  {img_name}: best T={info['best_T']:2d}  "
              f"PSNR={info['psnr']:.4f}  SSIM={info['ssim']:.4f}")

    # Save
    json_path = os.path.join(ADAPT_DIR, f"adaptive_stages_L{L}.json")
    with open(json_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\n  → {json_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--ckpt_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--test_dir", default=CLEAN_TEST_DIR)
    args = parser.parse_args()

    run_adaptive_study(L=args.L, ckpt_dir=args.ckpt_dir,
                        test_dir=args.test_dir)


if __name__ == "__main__":
    main()
