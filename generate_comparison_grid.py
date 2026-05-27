"""
generate_comparison_grid.py — Per-image model comparison for L=1 & L=10.
Shows: Noisy | Learned TNRD | NCTDN | PDE Baseline | Ground Truth
Each panel annotated with PSNR value on the image.
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
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, TAU, EMBED_DIM, NUM_NOISE_LEVELS
from models.network import InertialTNRDNetwork
from models.noise_conditional_network import NoiseConditionalTNRDNetwork
from utils.noise import add_gamma_noise
from evaluate import run_pde_baseline

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "comparison", "per_model_L1")
os.makedirs(OUT, exist_ok=True)
device = DEVICE

test_paths = sorted(glob.glob(os.path.join(CLEAN_TEST_DIR, "*.png")))[:12]
names = [os.path.splitext(os.path.basename(p))[0] for p in test_paths]

# ── Load models ──────────────────────────────────────────────────────────────
def load_learned(L):
    path = os.path.join(CHECKPOINT_DIR, f"model_L{L}_final.pth")
    sd = torch.load(path, map_location="cpu")
    ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    m = InertialTNRDNetwork(
        num_stages=ns, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True, device=device,
    ).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def load_nctdn():
    path = os.path.join(CHECKPOINT_DIR, "nctdn_model_mixed_final.pth")
    sd = torch.load(path, map_location="cpu")
    ns = max(int(k.split('.')[1]) for k in sd if k.startswith('stages.')) + 1
    m = NoiseConditionalTNRDNetwork(
        num_stages=ns, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
        K_thresh=K, num_centers=RBF_NUM_CENTERS, embed_dim=EMBED_DIM,
        num_noise_levels=NUM_NOISE_LEVELS, device=device,
    ).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

print("Loading models...")
learned = load_learned(1)
nctdn = load_nctdn()
print("Models loaded.")

for idx, img_path in enumerate(test_paths):
    name = names[idx]
    img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
    clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    noisy = add_gamma_noise(clean.clone(), 1).clip(0, 255)
    noisy_np = noisy.cpu().squeeze().numpy()
    clean_np = img

    with torch.no_grad():
        out_learned, _ = learned(noisy.to(device))
        out_learned = out_learned.clamp(0, 255).cpu().squeeze().numpy()

        out_nctdn, _ = nctdn(noisy.to(device), L=1)
        out_nctdn = out_nctdn.clamp(0, 255).cpu().squeeze().numpy()

    out_pde = run_pde_baseline(
        torch.from_numpy(noisy_np.copy()).unsqueeze(0).unsqueeze(0).to(device),
        gamma=GAMMA_INERTIA, tau=TAU, nu=NU, K_thresh=K, sigma=SIGMA_SMOOTH)
    out_pde = out_pde.cpu().squeeze().numpy().astype(np.float32)

    # PSNR values
    def calc_psnr(im):
        return 10 * np.log10(255**2 / np.mean((im - clean_np)**2 + 1e-10))

    ps_noisy = calc_psnr(noisy_np)
    ps_learned = calc_psnr(out_learned)
    ps_nctdn = calc_psnr(out_nctdn)
    ps_pde = calc_psnr(out_pde)

    # ── 5-panel comparison ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    panels = [
        (noisy_np, f"Noisy\n{ps_noisy:.2f} dB"),
        (out_learned, f"Learned TNRD\n{ps_learned:.2f} dB"),
        (out_nctdn, f"NCTDN\n{ps_nctdn:.2f} dB"),
        (out_pde, f"PDE Baseline\n{ps_pde:.2f} dB"),
        (clean_np, "Ground Truth\n---"),
    ]
    for ax, (im, label) in zip(axes, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")
    fig.suptitle(f"{name} — L=1 Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT, f"{name}_L1_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{idx+1}/12] {name}: Learned={ps_learned:.2f}dB  NCTDN={ps_nctdn:.2f}dB  PDE={ps_pde:.2f}dB  Noisy={ps_noisy:.2f}dB")
    torch.cuda.empty_cache()

print(f"\nDone! All comparisons in {OUT}")
