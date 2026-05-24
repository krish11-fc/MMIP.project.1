#!/usr/bin/env python3
"""
run_all_studies.py — Run all analyses and studies for Project 8.

Usage:
    python run_all_studies.py                         # run everything
    python run_all_studies.py --skip-sweep             # skip hyperparam sweep
    python run_all_studies.py --L 1                    # L=1 only
    python run_all_studies.py --study influence        # specific study only

Studies:
    sweep       — Hyperparameter sweep (K, ν, τ, γ, σ) via PDE baseline
    influence   — Influence function φ_i(s) per stage vs PDE baseline
    inertia     — u_tt (inertial term) ablation study
    compare     — Model comparison (TNRD vs PDE vs Full-Learn vs TNRD-Log)
    all         — Everything above
"""
import os, sys, argparse, subprocess, time

ROOT = os.path.dirname(os.path.abspath(__file__))

STUDIES = {
    "sweep":     "hyperparam_sweep.py",
    "influence": "influence_study.py",
    "inertia":   "ablate_inertia.py",
    "compare":   "compare_models.py",
}


def run_script(name, script, extra_args=None):
    path = os.path.join(ROOT, script)
    if not os.path.exists(path):
        print(f"  ⚠ Script not found: {path}")
        return
    cmd = [sys.executable, path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*70}")
    print(f"  RUNNING: {name}  ({' '.join(cmd)})")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"\n  {status} {name} finished in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all Project 8 studies")
    parser.add_argument("--study", choices=list(STUDIES.keys()) + ["all"],
                        default="all", help="Which study to run (default: all)")
    parser.add_argument("--L", type=int, default=None,
                        help="Noise level (default: 1 for most studies)")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip hyperparameter sweep (takes ~2 min)")
    args = parser.parse_args()

    if args.study == "all":
        studies_to_run = list(STUDIES.keys())
    else:
        studies_to_run = [args.study]

    if args.skip_sweep and "sweep" in studies_to_run:
        studies_to_run.remove("sweep")
        print("  Skipping hyperparameter sweep (--skip-sweep)")

    extra = []
    if args.L is not None:
        extra = ["--L", str(args.L)]

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)

    for study_name in studies_to_run:
        script = STUDIES[study_name]
        run_script(study_name, script, extra)

    print(f"\n{'='*70}")
    print(f"  ALL STUDIES COMPLETE")
    print(f"  Outputs → {os.path.join(ROOT, 'outputs')}/")
    print(f"    sweep/            — hyperparameter sweep JSON")
    print(f"    influence_study/  — influence function plots")
    print(f"    inertia_ablation/ — inertial term ablation")
    print(f"    comparison/       — model comparison")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
