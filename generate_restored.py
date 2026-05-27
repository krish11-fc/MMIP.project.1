import os, sys, argparse
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


def load_image(path):
    img = Image.open(path).convert("L")
    return torch.from_numpy(np.array(img, dtype=np.float32))


def save_comparison(clean, noisy, restored, path, title=""):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, t in zip(axes,
                         [clean, noisy, restored],
                         ["Clean", "Noisy", "Restored"]):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--stages", type=int, default=10)
    args = parser.parse_args()
    L = args.L
    T = args.stages
    device = DEVICE

    test_dir = CLEAN_TEST_DIR
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "outputs", "comparison", f"restored_L{L}")
    os.makedirs(out_dir, exist_ok=True)

    test_images = sorted([f for f in os.listdir(test_dir)
                          if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))])
    if not test_images:
        print(f"No images found in {test_dir}")
        return

    filt = build_dct_filters(NUM_FILTERS, FILTER_SIZE, device)

    learned = InertialTNRDNetwork(
        num_stages=T, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)

    nctdn = NoiseConditionalTNRDNetwork(
        num_stages=T, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True,
        embed_dim=EMBED_DIM, num_noise_levels=NUM_NOISE_LEVELS, device=device,
    ).to(device)

    ckpt_learned = os.path.join(CHECKPOINT_DIR, f"model_L{L}_final.pth")
    ckpt_nctdn = os.path.join(CHECKPOINT_DIR, "nctdn_model_mixed_final.pth")

    if os.path.exists(ckpt_learned):
        sd = torch.load(ckpt_learned, map_location="cpu")
        n_stages = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
        if n_stages > 0 and n_stages < T:
            learned = InertialTNRDNetwork(
                num_stages=n_stages, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
                gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
                K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
            ).to(device)
        learned.load_state_dict(sd, strict=False)
        learned.eval()
        print(f"Loaded learned: {ckpt_learned} ({n_stages} stages)")
    else:
        print(f"No learned ckpt: {ckpt_learned}")
        learned = None

    if os.path.exists(ckpt_nctdn):
        nctdn.load_state_dict(torch.load(ckpt_nctdn, map_location="cpu"))
        nctdn.eval()
        print(f"Loaded NCTDN: {ckpt_nctdn}")
    else:
        print(f"No NCTDN ckpt: {ckpt_nctdn}")
        nctdn = None

    print(f"\nGenerating restored images for L={L} on {len(test_images)} images...")
    for idx, fname in enumerate(test_images):
        clean = load_image(os.path.join(test_dir, fname))
        h, w = clean.shape
        inp = clean.unsqueeze(0).unsqueeze(0).to(device)
        noisy = add_gamma_noise(inp, L).clip(0, 255)

        clean_np = clean.cpu().squeeze().numpy()
        noisy_np = noisy.cpu().squeeze().numpy()
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes[0, 0].imshow(clean_np, cmap="gray", vmin=0, vmax=255)
        axes[0, 0].set_title("Clean")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(noisy_np, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title(f"Noisy L={L}")
        axes[0, 1].axis("off")

        with torch.no_grad():
            for col, (model, name, is_nctdn) in enumerate(
                [(learned, "Learned TNRD", False), (nctdn, "NCTDN", True)], start=2
            ):
                if model is None:
                    axes[0, col].axis("off")
                    axes[1, col].axis("off")
                    continue
                if is_nctdn:
                    out, _ = model(inp, L=L)
                else:
                    out, _ = model(inp)
                out_np = out.clamp(0, 255).cpu().squeeze().numpy()
                psnr = 10 * np.log10(255**2 / np.mean((out_np - clean_np)**2 + 1e-10))
                diff = np.abs(out_np - clean_np)
                axes[0, col].imshow(out_np, cmap="gray", vmin=0, vmax=255)
                axes[0, col].set_title(f"{name}\nPSNR={psnr:.1f}dB")
                axes[0, col].axis("off")
                axes[1, col].imshow(diff, cmap="hot", vmin=0, vmax=50)
                axes[1, col].set_title("|Error|")
                axes[1, col].axis("off")

        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        fig.suptitle(f"{fname}  (L={L})", fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_L{L}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{idx+1}/{len(test_images)}] {fname} -> {out_path}")

    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
