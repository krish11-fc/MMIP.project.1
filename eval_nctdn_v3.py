import os, sys, glob, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, CLEAN_VAL_DIR, NUM_FILTERS, FILTER_SIZE
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, EMBED_DIM, NUM_NOISE_LEVELS
from models.noise_conditional_network_v2 import NoiseConditionalTNRDNetworkV2
from utils.noise import add_gamma_noise
from utils.metrics import psnr, ssim

device = DEVICE

model = NoiseConditionalTNRDNetworkV2(10, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, True, EMBED_DIM, NUM_NOISE_LEVELS, device).to(device)
ckpt = os.path.join(CHECKPOINT_DIR, "nctdn_v2_model_mixed_final.pth")
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
print("Model loaded.")

@torch.no_grad()
def eval_on(paths, L, tag=""):
    ps, ss = [], []
    for p in paths:
        img = np.array(Image.open(p).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
        out, _ = model(noisy.to(device), L=L)
        out = out.clamp(0, 255).cpu()
        ps.append(psnr(out, clean).item())
        ss.append(ssim(out, clean))
    m, s = np.mean(ps), np.std(ps)
    sm, ss_s = np.mean(ss), np.std(ss)
    print(f"  L={L:<3}  PSNR={m:.2f} ± {s:.2f}  SSIM={sm:.4f} ± {ss_s:.4f}  [{tag}]")
    return {"psnr": round(m,2), "psnr_std": round(s,2), "ssim": round(sm,4), "ssim_std": round(ss_s,4)}

set12 = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "Set12", "*.png")))
bsd68 = sorted(glob.glob(os.path.join(CLEAN_VAL_DIR, "*.png")))[:68]
print(f"\nSet12: {len(set12)}, BSD68: {len(bsd68)}")

results = {}
for ds_name, paths in [("Set12", set12), ("BSD68", bsd68)]:
    print(f"\n--- {ds_name} ---")
    results[ds_name] = {}
    for L in [1, 3, 5, 10, 33]:
        results[ds_name][L] = eval_on(paths, L, ds_name)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "nctdn_v3_eval.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nDone! Results saved.")
