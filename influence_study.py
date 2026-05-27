"""
influence_study.py — Compare learned RBF influence functions φ_i(s) per stage
against the PDE baseline's influence (Perona-Malik edge-stopping c(s) and
gray-level indicator b(u)·c(|∇u|)).

Key question: What patterns do the learned φ_i(s) learn across stages?
- Early stages: learned RBF approximates PM? Noise removal?
- Late stages: learned RBF refines edges?
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NUM_STAGES, NUM_FILTERS, FILTER_SIZE,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS,
    CHECKPOINT_DIR,
)
from models import InertialTNRDNetwork

STUDY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "influence_study")
os.makedirs(STUDY_DIR, exist_ok=True)


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


def load_model(L, ckpt_dir, num_stages=None, use_g_func=True, device=DEVICE):
    gs = "" if use_g_func else "_nog"
    path, actual_stages = _find_best_ckpt(ckpt_dir, L, gs)
    if num_stages is None:
        num_stages = actual_stages
    model = InertialTNRDNetwork(
        num_stages=num_stages, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=use_g_func,
        device=device,
    ).to(device)
    if path and os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd)
        print(f"  Loaded: {path}  (stages={num_stages})")
    else:
        print(f"  WARNING: no checkpoint found for L={L}, using untrained model")
    return model


def plot_influence_functions_per_stage(model, L, save_dir, n_filters=8):
    """
    Plot φ_i(s) for each stage, all filters overlaid.
    Also overlay the PDE reference: Perona-Malik c(s) = 1/(1+s²) and
    the composite g(s) = b(s)·c(s) for typical image values.

    The PDE baseline's edge-stopping influence is:
        c(s) = 1/(1 + (s/K)²)
    where s = |∇u_ξ|.

    The learned model's influence per filter is:
        φ_i(r_i)  where r_i = K_i * u (filter response)
    
    These operate on different domains (gradient magnitude vs DCT responses),
    so we overlay Perona-Malik as a reference shape for comparison.
    """
    s = torch.linspace(-10, 10, 400, device=next(model.parameters()).device)
    s_4d = s.reshape(1, 1, 1, -1)

    # Perona-Malik reference: φ_PM(s) = 2s/(1+s²)
    pm_ref = (2.0 * s / (1.0 + s ** 2)).detach().cpu().numpy()
    s_np = s.detach().cpu().numpy()

    # PDE baseline edge-stopping: c(s) = 1/(1+(s/K)²)
    c_edge = (1.0 / (1.0 + (s / K) ** 2)).detach().cpu().numpy()

    T = len(model.stages)
    ncols = min(4, T)
    nrows = (T + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten() if T > 1 else [axes]

    for t in range(T):
        ax = axes[t]
        phi = model.stages[t].phi
        n_show = min(n_filters, phi.num_filters)

        with torch.no_grad():
            for i in range(n_show):
                y = phi(s_4d, filter_idx=i).squeeze().detach().cpu().numpy()
                ax.plot(s_np, y, linewidth=0.7, alpha=0.6,
                        label=f"φ_{i+1}" if t == 0 else None)

        ax.plot(s_np, pm_ref, "k--", linewidth=1.5, label="PM ref" if t == 0 else None)
        ax.plot(s_np, c_edge, "r:", linewidth=1.5, label="c(|∇|) ref" if t == 0 else None)
        ax.set_xlabel("s (filter response)")
        ax.set_ylabel("φ(s)")
        ax.set_title(f"Stage {t+1}  λ={model.stages[t].lambda_t.item():.4f}")
        ax.grid(alpha=0.3)
        if t == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide unused subplots
    for t in range(T, len(axes)):
        axes[t].set_visible(False)

    fig.suptitle(f"Learned RBF Influence Functions per Stage  (L={L}, T={T})", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, f"influence_per_stage_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, T))

    for fi in range(min(8, model.stages[0].phi.num_filters)):
        ax = axes[fi]
        with torch.no_grad():
            for t in range(T):
                y = model.stages[t].phi(s_4d, filter_idx=fi).squeeze().detach().cpu().numpy()
                ax.plot(s_np, y, color=colors[t], linewidth=1.0, alpha=0.8,
                        label=f"Stage {t+1}" if fi == 0 else None)
        ax.plot(s_np, pm_ref, "k--", linewidth=1.0, alpha=0.5, label="PM ref" if fi == 0 else None)
        ax.set_title(f"Filter {fi+1}")
        ax.grid(alpha=0.3)
        if fi == 0:
            ax.legend(fontsize=6, loc="best")

    fig.suptitle(f"Influence Evolution Across Stages per Filter  (L={L})", fontsize=13)
    fig.tight_layout()
    path = os.path.join(save_dir, f"influence_per_filter_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, ax in enumerate(axes):
        # Show weights for a few filters
        n_filt_show = 8
        w_all = []
        for t in range(T):
            w = model.stages[t].phi.weights[:n_filt_show].detach().cpu().numpy()
            w_all.append(w)
        w_stack = np.stack(w_all, axis=0)  # (T, n_filt_show, C)
        if idx == 0:
            im = ax.imshow(w_stack.mean(axis=1), aspect="auto", cmap="RdBu_r",
                           vmin=-1, vmax=1)
            ax.set_ylabel("Stage")
            ax.set_xlabel("RBF centre index")
            ax.set_title("Mean φ weights (avg over filters)")
        else:
            centres = model.stages[0].phi.centres.detach().cpu().numpy()
            diff = w_stack.std(axis=1)  # (T, C) std across filters
            im = ax.imshow(diff, aspect="auto", cmap="YlOrRd")
            ax.set_ylabel("Stage")
            ax.set_xlabel("RBF centre index")
            ax.set_title("Std of φ weights across filters")
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"RBF Weight Dynamics Across Stages  (L={L})", fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, f"rbf_weight_heatmap_L{L}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_pde_baseline_influence(save_dir):
    """Plot the PDE baseline's influence function components."""
    s = np.linspace(0, 500, 1000)

    # Edge-stopping: c(s) = 1/(1 + (s/K)²)
    K_val = K
    c = 1.0 / (1.0 + (s / K_val) ** 2)

    # Gray-level indicator (for typical M=200 for [0,255]): b(s) = 2s/(M+s)
    M_val = 200.0
    nu_val = NU
    b = (2.0 * s ** nu_val) / (M_val ** nu_val + s ** nu_val)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(s, c, "b-", linewidth=2)
    axes[0].axvline(K_val, color="r", linestyle="--", alpha=0.5, label=f"K={K_val}")
    axes[0].set_xlabel("|∇u_ξ| (gradient magnitude)")
    axes[0].set_ylabel("c(|∇u_ξ|)")
    axes[0].set_title("Edge-stopping: c(s) = 1/(1+(s/K)²)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(s, b, "g-", linewidth=2)
    axes[1].set_xlabel("u_ξ (smoothed intensity)")
    axes[1].set_ylabel("b(u_ξ)")
    axes[1].set_title(f"Gray-level: b(u) = 2u^ν/(M^ν+u^ν), ν={nu_val}")
    axes[1].grid(alpha=0.3)

    axes[2].plot(s, b * c, "purple", linewidth=2, label=f"g = b·c")
    axes[2].set_xlabel("s")
    axes[2].set_ylabel("g(s)")
    axes[2].set_title("Composite: g(u_ξ, |∇u_ξ|) = b · c")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle("PDE Baseline Influence Function Components  (Majee 2020)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(save_dir, "pde_baseline_influence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")

    s2 = np.linspace(-10, 10, 400)
    pm = 2.0 * s2 / (1.0 + s2 ** 2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s2, pm, "k-", linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("s (filter response)")
    ax.set_ylabel("φ_PM(s)")
    ax.set_title("Perona-Malik Influence Function  φ(s) = 2s/(1+s²)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(save_dir, "perona_malik_influence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--ckpt_dir", default=CHECKPOINT_DIR)
    parser.add_argument("--stages", type=int, default=NUM_STAGES)
    args = parser.parse_args()

    os.makedirs(STUDY_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  INFLUENCE FUNCTION STUDY  L={args.L}")
    print(f"{'='*60}")

    # PDE baseline influence components
    print("\n  Plotting PDE baseline influence...")
    plot_pde_baseline_influence(STUDY_DIR)

    # Learned model influence per stage
    print(f"\n  Loading learned model (L={args.L})...")
    model = load_model(args.L, args.ckpt_dir, num_stages=None, device=DEVICE)
    model.eval()
    model.print_param_summary()

    print(f"\n  Plotting learned influence functions...")
    plot_influence_functions_per_stage(model, args.L, STUDY_DIR)

    print(f"\n  All plots → {STUDY_DIR}/")


if __name__ == "__main__":
    main()
