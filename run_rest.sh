#!/usr/bin/env bash
set -euo pipefail
ROOT="/mnt/c/SEM-6/MMIP/project"
LOG="$ROOT/outputs/pipeline_rest.log"
L_VALS=(1)
cd "$ROOT"

# Wait for NCTDN to finish
while screen -ls 2>/dev/null | grep -q nctdn; do sleep 60; done

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  [7/11] Compare all models" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
python3 -u compare_models.py --L "${L_VALS[@]}" 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  [8/11] Optimal stopping analysis" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
for L in "${L_VALS[@]}"; do
    python3 -u optimal_stopping.py --L "$L" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  [9/11] Inertia (u_tt) ablation study" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
for L in "${L_VALS[@]}"; do
    python3 -u ablate_inertia.py --L "$L" 2>&1 | head -60 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  [10/11] Influence function study" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
for L in "${L_VALS[@]}"; do
    python3 -u influence_study.py --L "$L" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
echo "  [11/11] Adaptive stage selection" | tee -a "$LOG"
echo "════════════════════════════════════════════════════════════" | tee -a "$LOG"
for L in "${L_VALS[@]}"; do
    python3 -u adaptive_stages.py --L "$L" 2>&1 | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  PIPELINE COMPLETE" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
