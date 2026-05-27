import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch



def plot_loss_curve(
    train_losses: list,
    stage: int,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(train_losses) + 1), train_losses, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Stage {stage} — Training Loss")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_psnr_curve(
    train_psnrs: list,
    val_psnrs:   list,
    stage:       int,
    save_path:   str,
) -> None:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    epochs = range(1, len(train_psnrs) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, train_psnrs, label="Train PSNR", color="steelblue", linewidth=1.5)
    ax.plot(epochs, val_psnrs,   label="Val PSNR",   color="darkorange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"Stage {stage} — PSNR")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# Influence functions  φ_i(s)

def plot_influence_functions(
    phi,           # RBFInfluenceFunction module
    save_path: str,
    n_show: int = 8,
) -> None:
    """
    Plot learned RBF influence functions φ_i(s) for the first `n_show` filters.
    Also overlays the Perona-Malik reference curve 2s/(1+s²).
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    dev = phi.centres.device
    s_vals = torch.linspace(-10, 10, 400, device=dev, dtype=torch.float32)
    s_4d = s_vals.reshape(1, 1, 1, -1)

    fig, ax = plt.subplots(figsize=(7, 4))
    pm_ref = (2.0 * s_vals / (1.0 + s_vals ** 2)).detach().cpu().numpy()
    s_np = s_vals.detach().cpu().numpy()
    ax.plot(s_np, pm_ref, "k--", linewidth=1.2, label="Perona-Malik ref")

    n_show = min(n_show, phi.num_filters)
    with torch.no_grad():
        for i in range(n_show):
            y = phi(s_4d, filter_idx=i).squeeze().detach().cpu().numpy()
            ax.plot(s_np, y, linewidth=0.9, alpha=0.7,
                    label=f"φ_{i+1}" if i < 4 else None)

    ax.set_xlabel("s  (filter response)")
    ax.set_ylabel("φ(s)")
    ax.set_title("Learned influence functions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# Denoising result panels

def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().squeeze().numpy().clip(0, 255)


def plot_denoising_result(
    u_gt:       torch.Tensor,
    f:          torch.Tensor,
    u_pred:     torch.Tensor,
    psnr_noisy: float,
    psnr_pred:  float,
    ssim_noisy: float,
    ssim_pred:  float,
    save_path:  str,
    title:      str = "",
) -> None:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    imgs   = [_to_np(u_gt),   _to_np(f),        _to_np(u_pred)]
    titles = [
        "Ground truth",
        f"Noisy\nPSNR={psnr_noisy:.2f} SSIM={ssim_noisy:.4f}",
        f"Restored\nPSNR={psnr_pred:.2f} SSIM={ssim_pred:.4f}",
    ]
    for ax, img, t in zip(axes, imgs, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_stage_outputs(
    stage_outputs: list,
    u_gt:          torch.Tensor,
    f:             torch.Tensor,
    save_path:     str,
) -> None:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    T = len(stage_outputs)
    fig, axes = plt.subplots(1, T + 2, figsize=(3 * (T + 2), 3))
    axes[0].imshow(_to_np(f),    cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Noisy", fontsize=8)
    axes[0].axis("off")
    for t, so in enumerate(stage_outputs):
        axes[t + 1].imshow(_to_np(so), cmap="gray", vmin=0, vmax=255)
        axes[t + 1].set_title(f"Stage {t+1}", fontsize=8)
        axes[t + 1].axis("off")
    axes[-1].imshow(_to_np(u_gt), cmap="gray", vmin=0, vmax=255)
    axes[-1].set_title("GT", fontsize=8)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# Ablation charts

def plot_ablation_bar(
    results:   dict,
    metric:    str,
    xlabel:    str,
    title:     str,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    labels = list(results.keys())
    values = [results[k][metric] for k in labels]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4))
    bars = ax.bar(labels, values, color="steelblue", alpha=0.85)
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def save_results_table(
    all_results: dict,
    save_path:   str,
) -> None:
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    lines = [
        f"{'Method':<30}  {'PSNR (dB)':>10}  {'SSIM':>8}",
        "-" * 54,
    ]
    for method, res in all_results.items():
        psnr_v = res.get("psnr_mean", float("nan"))
        ssim_v = res.get("ssim_mean", float("nan"))
        lines.append(f"{method:<30}  {psnr_v:>10.4f}  {ssim_v:>8.4f}")
    with open(save_path, "w") as f:
        f.write("\n".join(lines) + "\n")
