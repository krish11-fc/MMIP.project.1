import os, sys
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, CLEAN_TEST_DIR, NUM_STAGES
from models.network import InertialTNRDNetwork
from models.noise_conditional_network import NoiseConditionalTNRDNetwork
from config import (
    NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    RBF_NUM_CENTERS, EMBED_DIM, NUM_NOISE_LEVELS,
)
from utils.filters import build_dct_filters
from utils.noise import add_gamma_noise

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "comparison")
os.makedirs(OUT_DIR, exist_ok=True)

device = DEVICE
filt = build_dct_filters(NUM_FILTERS, FILTER_SIZE, device)

test_dir = CLEAN_TEST_DIR
img_path = os.path.join(test_dir, "01.png")
img = Image.open(img_path).convert("L")
clean = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0)

for L in [1, 10]:
    inp = add_gamma_noise(clean.clone(), L).clip(0, 255)
    noisy_np = inp.cpu().squeeze().numpy()
    clean_np = clean.cpu().squeeze().numpy()

    ckpt = os.path.join(CHECKPOINT_DIR, f"model_L{L}_final.pth")
    try:
        sd = torch.load(ckpt, map_location="cpu")
        n_stages = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    except:
        n_stages = 5
    model = InertialTNRDNetwork(
        num_stages=n_stages, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()

    with torch.no_grad():
        out, all_stages = model(inp.to(device))
        out = out.clamp(0, 255).cpu().squeeze().numpy()

    # Single comparison: noisy vs restored
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, im, t in zip(axes, [clean_np, noisy_np, out],
                         ["Ground Truth", f"Noisy L={L}", "Restored"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis("off")
    psnr_val = 10 * np.log10(255**2 / np.mean((out - clean_np)**2 + 1e-10))
    fig.suptitle(f"L={L} — Learned TNRD (PSNR={psnr_val:.2f} dB)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"learned_test_000_L{L}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Stage progression (L=1 only)
    if L == 1:
        fig, axes = plt.subplots(1, n_stages + 2, figsize=(4*(n_stages+2), 4))
        axes[0].imshow(noisy_np, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title(f"Noisy L={L}")
        axes[0].axis("off")
        for t in range(n_stages):
            stg = all_stages[t].cpu().squeeze().numpy()
            ps = 10 * np.log10(255**2 / np.mean((stg - clean_np)**2 + 1e-10))
            axes[t+1].imshow(stg, cmap="gray", vmin=0, vmax=255)
            axes[t+1].set_title(f"Stage {t+1}\n{ps:.1f} dB")
            axes[t+1].axis("off")
        axes[n_stages+1].imshow(clean_np, cmap="gray", vmin=0, vmax=255)
        axes[n_stages+1].set_title("Ground Truth")
        axes[n_stages+1].axis("off")
        fig.suptitle(f"Stage Progression — L={L} ({n_stages} stages)", fontsize=14)
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"learned_test_000_L{L}_stages.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")

print("Done")
