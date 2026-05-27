import os, sys, glob, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINT_DIR, NUM_FILTERS, FILTER_SIZE
from config import GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, TAU
from config import EMBED_DIM, NUM_NOISE_LEVELS, CLEAN_VAL_DIR
from models import InertialTNRDNetwork, FullLearnInertialTNRDNetwork, TNRDLogNetwork
from models.noise_conditional_network import NoiseConditionalTNRDNetwork
from utils.noise import add_gamma_noise
from utils.metrics import psnr, ssim

device = DEVICE
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "bsd68_eval")
os.makedirs(results_dir, exist_ok=True)

def load_ckpt(ckpt_dir, name_suffix):
    path = os.path.join(ckpt_dir, name_suffix)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None

def load_learned(L):
    sd = load_ckpt(CHECKPOINT_DIR, f"model_L{L}_final.pth")
    if sd is None: return None
    ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    m = InertialTNRDNetwork(ns, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, True, device).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def load_fulllearn(L):
    try:
        sd = load_ckpt(CHECKPOINT_DIR, f"model_L{L}_fulllearn_final.pth")
        if sd is None: return None
        ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
        m = FullLearnInertialTNRDNetwork(ns, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, NU, K, RBF_NUM_CENTERS, True, device).to(device)
        m.load_state_dict(sd, strict=False)
        m.eval()
        return m
    except Exception as e:
        print(f"  [Full-Learn L={L} load error: {e}]")
        return None

def load_tnrdlog(L):
    sd = load_ckpt(CHECKPOINT_DIR, f"model_L{L}_tnrdlog_final.pth")
    if sd is None: return None
    ns = sum(1 for k in sd if k.startswith("stages.") and ".Ki" in k)
    m = TNRDLogNetwork(ns, NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS, device).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

def load_nctdn():
    sd = load_ckpt(CHECKPOINT_DIR, "nctdn_model_mixed_final.pth")
    if sd is None: return None
    ns = max(int(k.split('.')[1]) for k in sd if k.startswith('stages.')) + 1
    m = NoiseConditionalTNRDNetwork(ns, NUM_FILTERS, FILTER_SIZE, GAMMA_INERTIA, SIGMA_SMOOTH, NU, K, RBF_NUM_CENTERS, True, EMBED_DIM, NUM_NOISE_LEVELS, device).to(device)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m

@torch.no_grad()
def eval_model(model, loader, L, nctdn=False):
    ps, ss = [], []
    for clean, noisy in loader:
        if nctdn:
            out, _ = model(noisy.to(device), L=L)
        else:
            out, _ = model(noisy.to(device))
        out = out.clamp(0, 255).cpu()
        ps.append(psnr(out, clean).item())
        ss.append(ssim(out, clean).item())
    return np.mean(ps), np.std(ps), np.mean(ss), np.std(ss)

test_paths = sorted(glob.glob(os.path.join(CLEAN_VAL_DIR, "*.png")))[:68]
print(f"Found {len(test_paths)} BSD68 images")

for L in [1, 10]:
    print(f"\n{'='*60}")
    print(f"  BSD68 EVALUATION  L={L}")
    print(f"{'='*60}")
    
    images = []
    for p in test_paths:
        img = np.array(Image.open(p).convert("L"), dtype=np.float32)
        clean = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        noisy = add_gamma_noise(clean.clone(), L).clip(0, 255)
        images.append((clean, noisy))
    
    models = {}
    models["Learned TNRD"] = load_learned(L)
    models["Full-Learn"] = load_fulllearn(L)
    models["TNRD-Log"] = load_tnrdlog(L)
    models["NCTDN"] = ("nctdn", True) if load_nctdn() else None
    
    for name, model in sorted(models.items()):
        if model is None:
            print(f"  {name:<20}: NO CKPT")
            continue
        
        if name == "NCTDN":
            m = load_nctdn()
            if m is None:
                print(f"  NCTDN: NO CKPT")
                continue
            ps_mean, ps_std, ss_mean, ss_std = eval_model(m, images, L, nctdn=True)
        else:
            ps_mean, ps_std, ss_mean, ss_std = eval_model(model, images, L)
        
        print(f"  {name:<20}: PSNR={ps_mean:.2f} ± {ps_std:.2f}  SSIM={ss_mean:.4f} ± {ss_std:.4f}")
    
    torch.cuda.empty_cache()

print("\nDone!")
