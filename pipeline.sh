#!/usr/bin/env bash
set -euo pipefail
# =============================================================================
# pipeline.sh — Complete training + evaluation pipeline for Project 8
#
# Original (4 variants):
#   TNRD, Learn-K, Full-Learn, TNRD-Log
#
# Extension (NCTDN):
#   Noise-Conditional Telegraph Diffusion Network (paper-level)
#
# Usage:
#   ./pipeline.sh                        # full pipeline
#   ./pipeline.sh --L 1                  # L=1 only
#   ./pipeline.sh --skip-train           # evaluation only
#   ./pipeline.sh --quick                # 5 stages, 50 epochs
#   ./pipeline.sh --nctdn-only           # train/eval NCTDN only
# =============================================================================

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

L_VALS=(1)
SKIP_TRAIN=false
NCTDN_ONLY=false
STAGES=10
EPOCHS=150
MULTI_GPU="--no-multi-gpu"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --L) L_VALS=("$2"); shift 2 ;;
        --both) L_VALS=(1 10); shift ;;
        --skip-train) SKIP_TRAIN=true; shift ;;
        --quick) STAGES=5; EPOCHS=50; shift ;;
        --nctdn-only) NCTDN_ONLY=true; shift ;;
        --multi-gpu) MULTI_GPU=""; shift ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "$ROOT/outputs"

echo "============================================================"
echo "  MMIP Project 8 — Pipeline"
echo "  L = ${L_VALS[*]}   Stages=$STAGES   Epochs=$EPOCHS"
echo "  NCTDN extension: $(if $NCTDN_ONLY; then echo 'ONLY'; else echo 'included'; fi)"
echo "  Skip train: $SKIP_TRAIN"
echo "============================================================"

# ── 1. Hyperparameter sweep ─────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  [1/11] Hyperparameter sweep (PDE baseline)"
echo "════════════════════════════════════════════════════════════"
for L in "${L_VALS[@]}"; do
    python3 hyperparam_sweep.py --L "$L" --max_images 4
done

if [ "$SKIP_TRAIN" = false ]; then

if [ "$NCTDN_ONLY" = false ]; then
    # ── 2. Train original Inertial TNRD ──────────────────────────
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [2/11] Train original Inertial TNRD"
    echo "════════════════════════════════════════════════════════════"
    python3 train.py --pipeline --epochs "$EPOCHS" $MULTI_GPU

    # ── 3. Train Learn-K variant (learnable filters) ─────────────────
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [3/11] Train Learn-K variant (filters + gamma)"
    echo "════════════════════════════════════════════════════════════"
    python3 train_learn_k.py --pipeline --epochs "$EPOCHS" $MULTI_GPU

    # ── 4. Train Finetuned variant ───────────────────────────────
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [4/11] Train Full-Learn variant"
    echo "════════════════════════════════════════════════════════════"
    python3 train_full_learn.py --pipeline --epochs "$EPOCHS"

    # ── 5. Train TNRD-Log variant ────────────────────────────────
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [5/11] Train TNRD-Log variant"
    echo "════════════════════════════════════════════════════════════"
    python3 train_tnrd_log.py --pipeline --epochs "$EPOCHS" $MULTI_GPU
fi

    # ── 6. Train NCTDN (paper extension — single mixed-noise model) ──
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  [6/11] Train NCTDN — single model for all noise levels"
    echo "════════════════════════════════════════════════════════════"
    python3 train_nctdn.py --stages 10 --epochs 200 $MULTI_GPU

fi  # SKIP_TRAIN

# ── 7. Compare all models on Set12 ─────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  [7/11] Compare all models (TNRD, Full, Fine, Log, NCTDN)"
echo "════════════════════════════════════════════════════════════"
python3 compare_models.py --L "${L_VALS[@]}"

# ── 8. Optimal stopping analysis ───────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  [8/11] Optimal stopping analysis"
echo "════════════════════════════════════════════════════════════"
for L in "${L_VALS[@]}"; do
    python3 optimal_stopping.py --L "$L"
done

# ── 9. Inertia (u_tt) ablation ─────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  [9/11] Inertia (u_tt) ablation study"
echo "════════════════════════════════════════════════════════════"
for L in "${L_VALS[@]}"; do
    python3 ablate_inertia.py --L "$L" 2>&1 | head -60 || echo "  had issues (see above)"
done

# ── 10. Influence function study ───────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  [10/11] Influence function study"
echo "════════════════════════════════════════════════════════════"
for L in "${L_VALS[@]}"; do
    python3 influence_study.py --L "$L"
done

# ── 11. Adaptive stage selection ──────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  [11/11] Adaptive stage selection"
echo "════════════════════════════════════════════════════════════"
for L in "${L_VALS[@]}"; do
    python3 adaptive_stages.py --L "$L"
done

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Original: TNRD, Learn-K, Full-Learn, TNRD-Log"
echo "  Extension: NCTDN (noise-conditional, FiLM, adaptive)"
echo "  Outputs:"
echo "    outputs/sweep/             — hyperparameter sweep"
echo "    outputs/comparison/        — model comparison tables"
echo "    outputs/inertia_ablation/  — u_tt ablation"
echo "    outputs/influence_study/   — influence function plots"
echo "    outputs/optimal_stop/      — stopping analysis"
echo "    outputs/adaptive_stages/   — per-image depth analysis"
echo "    outputs/plots/             — training curves"
echo "    checkpoints/               — all trained models"
echo "============================================================"
