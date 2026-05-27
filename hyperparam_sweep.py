"""
hyperparam_sweep.py — Grid search over fixed PDE parameters (K, ν, τ, γ, σ).
Uses fast PDE baseline (no training). Results → outputs/sweep/.
"""
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CLEAN_TEST_DIR
from dataset import make_test_loader
from utils.metrics import psnr
from evaluate import run_pde_baseline

SWEEP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sweep")
os.makedirs(SWEEP_DIR, exist_ok=True)


def mean_psnr_on_loader(model_fn, loader, device):
    psnr_list = []
    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f = f.to(device)
        u_pred = model_fn(f).clamp(0.0, 255.0)
        psnr_list.append(psnr(u_pred, u_gt))
    return float(np.mean(psnr_list)) if psnr_list else 0.0


def run_sweep(L=1, test_dir=None, device=DEVICE, max_images=4):
    if test_dir is None:
        test_dir = CLEAN_TEST_DIR

    print(f"{'='*70}")
    print(f"  HYPERPARAMETER SWEEP  L={L}  (PDE baseline, {max_images} images)")
    print(f"{'='*70}")

    # Grid ranges
    grids = {
        "K":     [32, 64, 128, 256, 512],
        "nu":    [0.5, 1.0, 2.0, 4.0],
        "tau":   [0.05, 0.1, 0.2, 0.4, 0.8],
        "gamma": [0.0, 0.25, 0.5, 1.0, 2.0, 5.0],
        "sigma": [0.5, 1.0, 2.0, 4.0],
    }
    defaults = {"K": 128.0, "nu": 1.0, "tau": 0.2, "gamma": 0.5, "sigma": 1.0}
    best = dict(defaults)

    # Mini loader for fast sweeps
    full_loader = make_test_loader(test_dir, L=L, seed=0)
    limited = []
    for i, (u_gt, f) in enumerate(full_loader):
        if i >= max_images:
            break
        limited.append((u_gt, f))

    class MiniLoader:
        def __init__(self, data): self.data = data
        def __iter__(self): return iter(self.data)
    mini_loader = MiniLoader(limited)

    results = {}

    # Sweep each parameter independently, keeping others at running-best
    sweep_order = ["K", "nu", "tau", "gamma", "sigma"]
    for param in sweep_order:
        print(f"\n  --- Sweep: {param} ---")
        param_results = {}
        for val in grids[param]:
            cfg = dict(best)
            cfg[param] = val
            def make_fn(c=cfg):
                return lambda f: run_pde_baseline(
                    f, gamma=c["gamma"], tau=c["tau"], nu=c["nu"],
                    K_thresh=c["K"], sigma=c["sigma"])
            psnr_val = mean_psnr_on_loader(make_fn(cfg), mini_loader, device)
            param_results[val] = psnr_val
            print(f"    {param}={val!s:>5}  → PSNR={psnr_val:.4f}")
        best_val = max(param_results, key=param_results.get)
        best[param] = best_val
        results[f"{param}_sweep"] = param_results
        print(f"    → Best {param}={best_val}")

    # Validate on full test set
    default_fn = lambda f: run_pde_baseline(
        f, gamma=defaults["gamma"], tau=defaults["tau"], nu=defaults["nu"],
        K_thresh=defaults["K"], sigma=defaults["sigma"])
    best_fn = lambda f: run_pde_baseline(
        f, gamma=best["gamma"], tau=best["tau"], nu=best["nu"],
        K_thresh=best["K"], sigma=best["sigma"])

    default_psnr = mean_psnr_on_loader(default_fn, full_loader, device)
    best_psnr = mean_psnr_on_loader(best_fn, full_loader, device)

    summary = {
        "L": L,
        "default": defaults,
        "best": best,
        "default_psnr": default_psnr,
        "best_psnr": best_psnr,
        "improvement_db": best_psnr - default_psnr,
        "full_results": results,
    }

    print(f"\n{'='*70}")
    print(f"  RESULTS  L={L}")
    print(f"  Default: K={defaults['K']}, ν={defaults['nu']}, τ={defaults['tau']}, "
          f"γ={defaults['gamma']}, σ={defaults['sigma']}")
    print(f"    → PSNR = {default_psnr:.4f} dB")
    print(f"  Best:   K={best['K']}, ν={best['nu']}, τ={best['tau']}, "
          f"γ={best['gamma']}, σ={best['sigma']}")
    print(f"    → PSNR = {best_psnr:.4f} dB")
    print(f"  Improvement: {best_psnr - default_psnr:+.4f} dB")
    print(f"{'='*70}")

    path = os.path.join(SWEEP_DIR, f"sweep_L{L}.json")
    with open(path, "w") as fp:
        json.dump(summary, fp, indent=2, default=str)
    print(f"\n  → {path}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=4)
    args = parser.parse_args()
    run_sweep(L=args.L, max_images=args.max_images)


if __name__ == "__main__":
    main()
