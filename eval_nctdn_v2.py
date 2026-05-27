import os, sys, glob, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, NUM_FILTERS, FILTER_SIZE
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, EMBED_DIM, NUM_NOISE_LEVELS, CLEAN_VAL_DIR
from models.noise_conditional_network_v2 import NoiseConditionalTNRDNetworkV2
from utils.noise import add_gamma_noise
from utils.metrics import psnr, ssim

device = DEVICE
results = {}

@torch.no_grad()
def eval_model(model, images, L):
    ps_vals, ss_vals = [], []
    for clean, noisy in images:
        out, _ = model(noisy.to(device), L=L)
        out = out.clamp(0, 255).cpu()
        ps_vals.append(psnr(out, clean).item())
        ss_vals.append(ssim(out, clean))
    return np.mean(ps_vals), np.std(ps_vals), np.mean(ss_vals), np.std(ss_vals)

def load_images(paths):
    images = []
    for p in paths:
        img = np.array(Image.open(p).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        images.append(clean)
    return images

print("Loading NCTDN v2...")
model = NoiseConditionalTNRDNetworkV2(
    num_stages=10, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
    gamma_inertia=GAMMA_INERTIA, sigma_smooth=SIGMA_SMOOTH, nu=NU,
    K_thresh=K, num_centers=RBF_NUM_CENTERS, use_g_func=True,
    embed_dim=EMBED_DIM, num_noise_levels=NUM_NOISE_LEVELS, device=device,
).to(device)
# Try current naming first, then fall back to legacy
ckpt_path = os.path.join(CHECKPOINT_DIR, "nctdn_v2_model_mixed_final_nctdn_v2.pth")
if not os.path.exists(ckpt_path):
    ckpt_path = os.path.join(CHECKPOINT_DIR, "nctdn_v2_model_mixed_final.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt)
model.eval()
print("Model loaded.\n")

# Set12
set12_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Set12")
set12_paths = sorted(glob.glob(os.path.join(set12_dir, "*.png")))
print(f"Set12: {len(set12_paths)} images")

# BSD68
bsd68_paths = sorted(glob.glob(os.path.join(CLEAN_VAL_DIR, "*.png")))[:68]
print(f"BSD68: {len(bsd68_paths)} images")

for dataset_name, paths in [("Set12", set12_paths), ("BSD68", bsd68_paths)]:
    clean_images = load_images(paths)
    print(f"\n{'='*60}")
    print(f"  {dataset_name}")
    print(f"{'='*60}")
    results[dataset_name] = {}
    for L in [1, 3, 5, 10, 33]:
        noisy_images = [(c, add_gamma_noise(c.clone(), L).clip(0, 255)) for c in clean_images]
        ps_mean, ps_std, ss_mean, ss_std = eval_model(model, noisy_images, L)
        results[dataset_name][L] = {
            "psnr_mean": round(ps_mean, 2),
            "psnr_std": round(ps_std, 2),
            "ssim_mean": round(ss_mean, 4),
            "ssim_std": round(ss_std, 4),
        }
        print(f"  L={L:<3}  PSNR={ps_mean:.2f} ± {ps_std:.2f}  SSIM={ss_mean:.4f} ± {ss_std:.4f}")

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "nctdn_v2_eval.json")
with open(save_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {save_path}")
print("Done!")
