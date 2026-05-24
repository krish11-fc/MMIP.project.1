"""
train_finetuned.py — Train original InertialTNRD with learnable PDE scalars.

Same greedy stage-wise training as train.py, but uses FinetunedInertialTNRDNetwork
where K, γ, τ, ν, σ are nn.Parameters (fine-tuned alongside RBF φ_i).

Filters stay frozen.  Only 5 scalars + RBF + λ are learned per stage.

Checkpoints saved with _finetuned suffix.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DEVICE, NUM_STAGES, BATCH_SIZE, PATCH_SIZE,
    NUM_EPOCHS, LR, LR_STEP, LR_GAMMA_SCHED, WEIGHT_DECAY, GRAD_CLIP,
    CLEAN_TRAIN_DIR, CLEAN_VAL_DIR,
    CHECKPOINT_DIR, PLOT_DIR,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS,
    RBF_CENTER_MIN, RBF_CENTER_MAX,
    DATALOADER_NUM_WORKERS, VAL_MAX_IMAGES,
    CURRICULUM_L1_SCHEDULE,
)
from models import FinetunedInertialTNRDNetwork
from dataset import make_train_loader, make_val_loader
from utils.visualization import plot_loss_curve, plot_psnr_curve, plot_influence_functions

_WARMUP_EPOCHS = 3
SUFFIX = "_finetuned"


def _psnr255(pred, target):
    with torch.no_grad():
        mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * float(np.log10(255.0 ** 2 / mse))


def _ssim255(pred, target):
    p = pred.detach().cpu().float().numpy()
    t = target.detach().cpu().float().numpy()
    if p.ndim == 4:
        scores = [sk_ssim(p[i, 0], t[i, 0], data_range=255.0) for i in range(p.shape[0])]
        return float(np.mean(scores))
    return float(sk_ssim(p.squeeze(), t.squeeze(), data_range=255.0))


def train_one_epoch(model, loader, optimizer, device, active_stages):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f = f.to(device)
        optimizer.zero_grad()
        u_pred, _ = model(f, active_stages=active_stages)
        loss = criterion(u_pred, u_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, active_stages):
    model.eval()
    psnr_list, ssim_list = [], []
    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f = f.to(device)
        u_pred, _ = model(f, active_stages=active_stages)
        u_pred = u_pred.clamp(0.0, 255.0)
        psnr_list.append(_psnr255(u_pred, u_gt))
        ssim_list.append(_ssim255(u_pred, u_gt))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def _set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_stage(model, stage_idx, train_loader, val_loader, device,
                num_epochs=NUM_EPOCHS, lr=LR, L=1,
                save_dir=CHECKPOINT_DIR, plot_dir=PLOT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    model.freeze_stages(up_to=stage_idx)
    model.unfreeze_stage(stage_idx)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)

    print(f"\n{'='*60}")
    print(f"  [Finetuned] Stage {stage_idx+1}/{model.T}  |  L={L}")
    print(f"  trainable params: {n_trainable:,}  |  epochs={num_epochs}")
    print(f"{'='*60}")

    optimizer = optim.Adam(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA_SCHED)
    active = stage_idx + 1

    history = {"train_loss": [], "val_psnr": [], "val_ssim": []}
    best_psnr = -1.0
    best_path = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        if epoch <= _WARMUP_EPOCHS:
            _set_lr(optimizer, lr * epoch / _WARMUP_EPOCHS)
        elif epoch == _WARMUP_EPOCHS + 1:
            _set_lr(optimizer, lr)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, active)
        val_psnr, val_ssim = validate(model, val_loader, device, active)

        if epoch > _WARMUP_EPOCHS:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # Log scalars every epoch
        params_str = ""
        stage = model.stages[stage_idx]
        params_str = (f"  γ={stage.gamma.item():.4f}  τ={stage.tau.item():.4f}  "
                      f"K={stage.K_thresh.item():.1f}  ν={stage.nu.item():.4f}  "
                      f"σ={stage.sigma.item():.4f}")

        print(f"  ep={epoch:3d}/{num_epochs}  loss={train_loss:.2f}  "
              f"val_psnr={val_psnr:.4f}  val_ssim={val_ssim:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"time={time.time()-t0:.1f}s{params_str}")

        # Save best checkpoint
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            save_path = os.path.join(
                save_dir,
                f"stage{stage_idx+1}_L{L}_best{SUFFIX}.pth")
            torch.save(model.state_dict(), save_path)
            best_path = save_path
            print(f"  *** New best val PSNR: {val_psnr:.4f} (ep {epoch}) → {save_path}")

        # Save last checkpoint
        last_path = os.path.join(
            save_dir,
            f"stage{stage_idx+1}_L{L}_last{SUFFIX}.pth")
        torch.save(model.state_dict(), last_path)

    print(f"  Stage {stage_idx+1} done.  Best val PSNR = {best_psnr:.4f} @ ep {best_epoch}")
    return history, best_path


def _find_resume_stage(L, num_stages, save_dir=CHECKPOINT_DIR):
    final_path = os.path.join(save_dir, f"model_L{L}_final{SUFFIX}.pth")
    if os.path.isfile(final_path):
        return None, None

    last_done = 0
    for i in range(1, num_stages + 1):
        ckpt = os.path.join(save_dir, f"stage{i}_L{L}_last{SUFFIX}.pth")
        if os.path.isfile(ckpt):
            last_done = i
        else:
            break

    next_stage_idx = last_done
    if next_stage_idx >= num_stages:
        return None, None

    if last_done == 0:
        return next_stage_idx, None

    ckpt_path = os.path.join(save_dir, f"stage{last_done}_L{L}_last{SUFFIX}.pth")
    return next_stage_idx, ckpt_path


def train(L=1, num_stages=NUM_STAGES, epochs=NUM_EPOCHS, workers=None,
          save_dir=CHECKPOINT_DIR, plot_dir=PLOT_DIR, resume=False):
    device = DEVICE

    if resume:
        start_stage, load_path = _find_resume_stage(L, num_stages, save_dir)
        if start_stage is None:
            print(f"\n  [L={L}] Already complete — skipping.")
            return None
        print(f"\n  [L={L}] Resuming from stage {start_stage+1}/{num_stages}")
    else:
        start_stage = 0
        load_path = None

    print("#" * 60)
    print(f"  FINETUNED TRAINING  L={L}  T={num_stages}")
    if resume:
        print(f"  RESUME from stage {start_stage+1}")
    print(f"  Learnable: K, γ, τ, ν, σ, φ_i (RBF), λ^t")
    print("#" * 60)

    model = FinetunedInertialTNRDNetwork(
        num_stages=num_stages, num_filters=NUM_FILTERS, filter_size=FILTER_SIZE,
        gamma_init=GAMMA_INERTIA, tau_init=0.2, nu_init=NU,
        K_init=K, sigma_init=SIGMA_SMOOTH,
        num_centers=RBF_NUM_CENTERS, use_g_func=True,
        device=device,
    ).to(device)

    if load_path is not None:
        sd = torch.load(load_path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device)
        print(f"  Loaded checkpoint: {load_path}")

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Params: {n_total:,} trainable  |  {n_fixed:,} fixed")
    for t in range(num_stages):
        s = model.stages[t]
        print(f"    Stage {t+1}: γ={s.gamma.item():.4f}  τ={s.tau.item():.4f}  "
              f"K={s.K_thresh.item():.1f}  ν={s.nu.item():.4f}  σ={s.sigma.item():.4f}")

    history_all = {}
    for stage_idx in range(start_stage, num_stages):
        if L == 1 and stage_idx < len(CURRICULUM_L1_SCHEDULE):
            L_curriculum = CURRICULUM_L1_SCHEDULE[stage_idx]
        else:
            L_curriculum = L

        print(f"\n  Curriculum L={L_curriculum} for stage {stage_idx+1}")
        train_loader_cur = make_train_loader(
            CLEAN_TRAIN_DIR, L=L_curriculum,
            batch_size=BATCH_SIZE, patch_size=PATCH_SIZE,
            num_workers=workers, max_images=None)
        val_loader_cur = make_val_loader(
            CLEAN_VAL_DIR, L=L_curriculum,
            batch_size=BATCH_SIZE, patch_size=PATCH_SIZE,
            seed=42, num_workers=workers,
            max_images=VAL_MAX_IMAGES)

        hist, best_path = train_stage(
            model, stage_idx, train_loader_cur, val_loader_cur, device,
            num_epochs=epochs, lr=LR, L=L_curriculum,
            save_dir=save_dir, plot_dir=plot_dir)
        history_all[stage_idx] = hist

    # Save final model
    final_path = os.path.join(save_dir, f"model_L{L}_final{SUFFIX}.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n  Final model → {final_path}")

    # Save training history
    hist_path = os.path.join(save_dir, f"training_history_L{L}{SUFFIX}.json")
    old_hist = {}
    if resume and os.path.isfile(hist_path):
        with open(hist_path) as fp:
            old_hist = json.load(fp)
    old_hist.update({str(k): v for k, v in history_all.items()})
    with open(hist_path, "w") as fp:
        json.dump(old_hist, fp, indent=2, default=str)
    print(f"  History → {hist_path}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--stages", type=int, default=NUM_STAGES)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--both", action="store_true",
                        help="Train both L=1 and L=10 sequentially")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run/resume all L values in sequence (1, 10)")
    args = parser.parse_args()

    if args.pipeline:
        for L in [1, 10]:
            ns = 10 if L == 1 else min(args.stages, 5)
            train(L=L, num_stages=ns, epochs=args.epochs,
                  workers=args.workers, resume=True)
    elif args.both:
        for L in [10, 1]:
            train(L=L, num_stages=min(args.stages, 5) if L == 10 else args.stages,
                  epochs=args.epochs, workers=args.workers, resume=args.resume)
    else:
        train(L=args.L, num_stages=args.stages,
              epochs=args.epochs, workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
