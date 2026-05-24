
import os
from pathlib import Path
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent

# ── Device ────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU required but not found. "
        "If you are on WSL2, run 'wsl --shutdown' from PowerShell and restart."
    )
DEVICE = torch.device("cuda")


def default_dataloader_workers() -> int:
    n = os.cpu_count() or 4
    return min(8, max(0, n - 1))


DATALOADER_NUM_WORKERS = default_dataloader_workers()

# ── Filter bank  k_i  ── FIXED, never learned ────────────────────────────────
NUM_FILTERS  = 48
FILTER_SIZE  = 7

# ── RBF influence function  φ_i  ── LEARNED per stage ────────────────────────
RBF_NUM_CENTERS  = 63
RBF_CENTER_MIN   = -300.0
RBF_CENTER_MAX   =  300.0

# ── PDE / architecture parameters ────────────────────────────────────────────
NUM_STAGES     = 10

# Default (conservative) PDE parameters — work well across noise levels.
TAU            = 0.2
GAMMA_INERTIA  = 0.5
SIGMA_SMOOTH   = 1.0
NU             = 1.0
K              = 128.0   # BUG 1 FIX — never change this back
EPSILON        = 1e-8

# ── OPTIMAL PDE PARAMETERS (from hyperparam_sweep.py) ────────────────────────
# For best results at a specific noise level, swap in these values.
#
# L=1  (heavy noise):  K=32,   ν=1.0, τ=0.2, γ=1.0, σ=1.0
#   → PDE baseline PSNR improves from 16.06 → 16.30 dB (+0.24 dB)
#
# L=10 (light noise):  K=512,  ν=0.5, τ=0.4, γ=2.0, σ=0.5
#   → PDE baseline PSNR improves from 16.68 → 22.41 dB (+5.73 dB!)
#
# Note: optimal parameters for the *learned* model may differ slightly
# because the RBF φ_i can adapt. Use these as a warm-start.
OPTIMAL_PARAMS = {
    1:  {"K": 32.0,  "NU": 1.0, "TAU": 0.2, "GAMMA": 1.0, "SIGMA": 1.0},
    10: {"K": 512.0, "NU": 0.5, "TAU": 0.4, "GAMMA": 2.0, "SIGMA": 0.5},
}

def get_optimal_params(L: int) -> dict:
    """Return optimal PDE parameters for noise level L."""
    return OPTIMAL_PARAMS.get(L, {"K": K, "NU": NU, "TAU": TAU,
                                   "GAMMA": GAMMA_INERTIA, "SIGMA": SIGMA_SMOOTH})

# ── Noise levels ──────────────────────────────────────────────────────────────
NOISE_LEVELS      = [1, 10]
NOISE_LEVELS_ALL  = [1, 3, 5, 10, 33]
DEFAULT_L         = 1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 8
PATCH_SIZE        = 64
# CHANGE 2: 100 → 150 epochs per stage for L=1 (model still learning at ep 83)
NUM_EPOCHS        = 150     # was 100
LR                = 1e-3
LR_STEP           = 50      # was 30 — scale with more epochs (halvings at ep50, ep100)
LR_GAMMA_SCHED    = 0.5
WEIGHT_DECAY      = 1e-5
GRAD_CLIP         = 1.0
VAL_MAX_IMAGES    = 20

# ── CHANGE 3: Curriculum schedule for L=1 ────────────────────────────────────
# Maps stage index (0-based) → noise level L to train that stage on.
# Earlier stages see easier noise (higher L) to learn good base diffusers.
# Later stages fine-tune on hard L=1 noise.
# For L=10 training: all stages use L=10 (no curriculum needed, converges well).
#
# Stage:  0    1    2    3    4    5    6    7    8    9
CURRICULUM_L1_SCHEDULE = [5,   5,   3,   3,   1,   1,   1,   1,   1,   1]
# Interpretation:
#   Stages 0-1: train on L=5  (mild-ish noise, learns basic smoothing fast)
#   Stages 2-3: train on L=3  (moderate noise, sharpens edge response)
#   Stages 4-9: train on L=1  (target noise, full fine-tuning)
#
# If you only have time for 5 stages, use:
# CURRICULUM_L1_SCHEDULE = [5, 3, 1, 1, 1]

# ── NCTDN (Noise-Conditional) extension ──────────────────────────────────────
NOISE_LEVELS_ALL  = [1, 3, 5, 10, 33]
NUM_NOISE_LEVELS  = max(NOISE_LEVELS_ALL) + 1
EMBED_DIM         = 16
ADAPTIVE_CURRICULUM_THRESHOLD = 0.3
DYNAMIC_STOP_THRESHOLD        = 0.01
WARMUP_EPOCHS     = 3
COMPARISON_DIR    = str(_PROJECT_ROOT / "outputs" / "comparison")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = str(_PROJECT_ROOT / "data")


def _nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


_TRAIN_LARGE = _PROJECT_ROOT / "data" / "BSD400_bsds500"
_TRAIN_SMALL = _PROJECT_ROOT / "data" / "BSD400"
CLEAN_TRAIN_DIR = str(_TRAIN_LARGE if _nonempty_dir(_TRAIN_LARGE) else _TRAIN_SMALL)

CLEAN_VAL_DIR = str(_PROJECT_ROOT / "data" / "BSD68")

_SET12     = _PROJECT_ROOT / "data" / "Set12"
_TEST_IMGS = _PROJECT_ROOT / "data" / "test_images"
CLEAN_TEST_DIR = str(_SET12 if _nonempty_dir(_SET12) else _TEST_IMGS)

SAR_TEST_DIR         = str(_PROJECT_ROOT / "data" / "SAR_images")
NOISY_TRAIN_DIR_TMPL = str(_PROJECT_ROOT / "data" / "noisy_train_L{}")
NOISY_VAL_DIR_TMPL   = str(_PROJECT_ROOT / "data" / "noisy_val_L{}")

CHECKPOINT_DIR  = str(_PROJECT_ROOT / "checkpoints")
PLOT_DIR        = str(_PROJECT_ROOT / "outputs" / "plots")
ABLATION_DIR    = str(_PROJECT_ROOT / "outputs" / "ablation")
RESULTS_DIR     = str(_PROJECT_ROOT / "outputs" / "results")
TABLES_DIR      = str(_PROJECT_ROOT / "outputs" / "tables")