"""
generate_all_visuals.py — All paper figures in one shot.
"""
import os, sys, glob
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, CLEAN_TEST_DIR, NUM_FILTERS, FILTER_SIZE
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS
from models.network import InertialTNRDNetwork
from utils.noise import add_gamma_noise

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "paper_figures")
os.makedirs(OUT, exist_ok=True)
device = DEVICE

# Get sorted test images
test_paths = sorted(glob.glob(os.path.join(CLEAN_TEST_DIR, "*.png")))
names = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]

def load_model(L):
    path = os.path.join(CHECKPOINT_DIR, f"model_L{L}_final.pth")
    sd = torch.load(path, map_location="cpu")
    ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    model = InertialTNRDNetwork(
        num_stages=ns, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, ns

# ── 1. Per-image side-by-side (clean | noisy | restored) ──────────────────────
print("=" * 60)
print("1. Per-image side-by-side results")
for L in [1, 10]:
    model, ns = load_model(L)
    for idx in range(min(12, len(test_paths))):
        path = test_paths[idx]
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
        with torch.no_grad():
            out, _ = model(noisy.to(device))
            out = out.clamp(0, 255).cpu().squeeze().numpy()
        noisy_np = noisy.cpu().squeeze().numpy()
        clean_np = img

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, im, t in zip(axes, [clean_np, noisy_np, out],
                             ["Ground Truth", f"Noisy L={L}", "Restored"]):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(t, fontsize=11)
            ax.axis("off")
        psnr_val = 10 * np.log10(255**2 / np.mean((out - clean_np)**2 + 1e-10))
        fig.suptitle(f"{names[idx]} — L={L} — {psnr_val:.2f} dB", fontsize=13)
        plt.tight_layout()
        fname = os.path.join(OUT, f"{names[idx]}_L{L}_result.png")
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")
    torch.cuda.empty_cache()

# ── 2. Stage progression (first 3 images, L=1 and L=10) ───────────────────
print("\n" + "=" * 60)
print("2. Stage progression")
for L in [1, 10]:
    model, ns = load_model(L)
    for idx in range(min(3, len(test_paths))):
        path = test_paths[idx]
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
        with torch.no_grad():
            _, stages = model(noisy.to(device))
        n_cols = ns + 2
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
        axes[0].imshow(noisy.cpu().squeeze(), cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Noisy", fontsize=9)
        axes[0].axis("off")
        for t in range(ns):
            stg = stages[t].cpu().squeeze().numpy()
            ps = 10 * np.log10(255**2 / np.mean((stg - img)**2 + 1e-10))
            axes[t + 1].imshow(stg, cmap="gray", vmin=0, vmax=255)
            axes[t + 1].set_title(f"Stage {t+1}\n{ps:.1f} dB", fontsize=8)
            axes[t + 1].axis("off")
        axes[ns + 1].imshow(img, cmap="gray", vmin=0, vmax=255)
        axes[ns + 1].set_title("Ground Truth", fontsize=9)
        axes[ns + 1].axis("off")
        fig.suptitle(f"{names[idx]} — L={L} ({ns} stages)", fontsize=14)
        plt.tight_layout()
        fname = os.path.join(OUT, f"{names[idx]}_L{L}_stages.png")
        fig.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")
    torch.cuda.empty_cache()

# ── 3. Best denoised 4×3 grid (all 12 Set12 images) ───────────────────────
print("\n" + "=" * 60)
print("3. Best denoised 4×3 grid")
for L in [1, 10]:
    model, ns = load_model(L)
    fig, axes = plt.subplots(4, 3, figsize=(10, 13))
    for idx in range(12):
        path = test_paths[idx]
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
        with torch.no_grad():
            out, _ = model(noisy.to(device))
            out = out.clamp(0, 255).cpu().squeeze().numpy()
        psnr_val = 10 * np.log10(255**2 / np.mean((out - img)**2 + 1e-10))
        ax = axes[idx // 3, idx % 3]
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{names[idx]} — {psnr_val:.1f} dB", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Learned TNRD Restored Results — L={L}", fontsize=14, y=1.01)
    plt.tight_layout()
    fname = os.path.join(OUT, f"all_restored_L{L}.png")
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")
    torch.cuda.empty_cache()

print("\nDone! All visual results in:", OUT)
