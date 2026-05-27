import os
import sys
import json
import time
import argparse
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
    CLEAN_TRAIN_DIR, CLEAN_VAL_DIR, NOISE_LEVELS,
    CHECKPOINT_DIR, PLOT_DIR,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS,
    RBF_CENTER_MIN, RBF_CENTER_MAX,
    DATALOADER_NUM_WORKERS, VAL_MAX_IMAGES,
    CURRICULUM_L1_SCHEDULE,   # NEW
)
from models import InertialTNRDNetwork
from dataset import make_train_loader, make_val_loader
from utils.visualization import plot_loss_curve, plot_psnr_curve, plot_influence_functions
from utils.losses import edge_preserving_loss

_WARMUP_EPOCHS = 3



def _psnr255(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * float(np.log10(255.0 ** 2 / mse))


def _ssim255(pred: torch.Tensor, target: torch.Tensor) -> float:
    p = pred.detach().cpu().float().numpy()
    t = target.detach().cpu().float().numpy()
    if p.ndim == 4:
        scores = [sk_ssim(p[i, 0], t[i, 0], data_range=255.0)
                  for i in range(p.shape[0])]
        return float(np.mean(scores))
    return float(sk_ssim(p.squeeze(), t.squeeze(), data_range=255.0))



def train_one_epoch(model, loader, optimizer, device, active_stages,
                    edge_weight=0.0) -> float:
    model.train()
    criterion  = nn.MSELoss()
    total_loss = 0.0

    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f    = f.to(device)
        optimizer.zero_grad()
        u_pred, _ = model(f, active_stages=active_stages)
        loss = criterion(u_pred, u_gt)
        if edge_weight > 0:
            loss = loss + edge_preserving_loss(u_pred, u_gt, weight=edge_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=GRAD_CLIP,
        )
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# Validation

@torch.no_grad()
def validate(model, loader, device, active_stages) -> tuple:
    model.eval()
    psnr_list, ssim_list = [], []

    for u_gt, f in loader:
        u_gt   = u_gt.to(device)
        f      = f.to(device)
        u_pred, _ = model(f, active_stages=active_stages)
        u_pred = u_pred.clamp(0.0, 255.0)
        psnr_list.append(_psnr255(u_pred, u_gt))
        ssim_list.append(_ssim255(u_pred, u_gt))

    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


# LR warmup helper

def _set_lr(optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# Train one stage (greedy) — UNCHANGED from original

def train_stage(
    model, stage_idx, train_loader, val_loader, device,
    num_epochs=NUM_EPOCHS, lr=LR, L=1,
    use_g_func=True, edge_weight=0.0,
    save_dir=CHECKPOINT_DIR,
    plot_dir=PLOT_DIR,
) -> dict:
    """
    Train exactly one stage (stage_idx), freezing all prior stages.
    Returns history dict with train_loss, train_psnr, val_psnr, val_ssim.
    (Unchanged from original — curriculum is handled in train_greedy.)
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    base = model.module if isinstance(model, nn.DataParallel) else model

    base.freeze_stages(up_to=stage_idx)
    base.unfreeze_stage(stage_idx)

    trainable   = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    g_suffix    = "" if use_g_func else "_nog"

    print(f"\n{'='*60}")
    print(f"  Stage {stage_idx+1}/{base.T}  |  L={L}  |  use_g={use_g_func}")
    print(f"  trainable params: {n_trainable:,}  |  K={K}  |  epochs={num_epochs}")
    print(f"  LR schedule: {lr} → ×0.5 @ ep{LR_STEP} → ×0.25 @ ep{LR_STEP*2}")
    print(f"  RBF centres: [{RBF_CENTER_MIN}, {RBF_CENTER_MAX}]  n={RBF_NUM_CENTERS}")
    print(f"{'='*60}")

    optimizer = optim.Adam(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA_SCHED)
    active    = stage_idx + 1

    history    = {"train_loss": [], "train_psnr": [], "val_psnr": [], "val_ssim": []}
    best_psnr  = -1.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        if epoch <= _WARMUP_EPOCHS:
            _set_lr(optimizer, lr * epoch / _WARMUP_EPOCHS)
        elif epoch == _WARMUP_EPOCHS + 1:
            _set_lr(optimizer, lr)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, active,
                                       edge_weight=edge_weight)
        train_psnr_val = 10.0 * np.log10(255.0 ** 2 / max(train_loss, 1e-10))

        # BUG 5 FIX: validate every epoch (was erroneously commented out)
        val_psnr_val, val_ssim_val = validate(model, val_loader, device, active)

        if epoch > _WARMUP_EPOCHS:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_psnr"].append(train_psnr_val)
        history["val_psnr"].append(val_psnr_val)
        history["val_ssim"].append(val_ssim_val)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Ep [{epoch:3d}/{num_epochs}]  "
              f"loss={train_loss:.4f}  "
              f"tPSNR≈{train_psnr_val:.2f}dB  "
              f"vPSNR={val_psnr_val:.2f}dB  "
              f"vSSIM={val_ssim_val:.4f}  "
              f"lr={cur_lr:.2e}  "
              f"({time.time()-t0:.1f}s)")

        if val_psnr_val > best_psnr:
            best_psnr  = val_psnr_val
            best_epoch = epoch
            base.save(os.path.join(
                save_dir, f"stage{stage_idx+1}_L{L}{g_suffix}_best.pth"))

    print(f"\n  Best val PSNR = {best_psnr:.2f} dB  at epoch {best_epoch}")
    base.save(os.path.join(save_dir, f"stage{stage_idx+1}_L{L}{g_suffix}_last.pth"))

    plot_loss_curve(
        history["train_loss"], stage=stage_idx + 1,
        save_path=os.path.join(plot_dir, f"loss_stage{stage_idx+1}_L{L}{g_suffix}.png"),
    )
    plot_psnr_curve(
        history["train_psnr"], history["val_psnr"],
        stage=stage_idx + 1,
        save_path=os.path.join(plot_dir, f"psnr_stage{stage_idx+1}_L{L}{g_suffix}.png"),
    )
    plot_influence_functions(
        base.stages[stage_idx].phi,
        save_path=os.path.join(plot_dir, f"phi_stage{stage_idx+1}_L{L}{g_suffix}.png"),
    )

    return history


# Full greedy training loop — WITH CURRICULUM for L=1

def _find_resume_stage(L, num_stages, save_dir, suffix):
    final_path = os.path.join(save_dir, f"model_L{L}{suffix}_final.pth")
    if os.path.isfile(final_path):
        return None, None

    last_done = 0
    for i in range(1, num_stages + 1):
        ckpt = os.path.join(save_dir, f"stage{i}_L{L}{suffix}_last.pth")
        if os.path.isfile(ckpt):
            last_done = i
        else:
            break

    next_stage_idx = last_done
    if next_stage_idx >= num_stages:
        return None, None

    if last_done == 0:
        return next_stage_idx, None

    ckpt_path = os.path.join(save_dir, f"stage{last_done}_L{L}{suffix}_last.pth")
    return next_stage_idx, ckpt_path


def train_greedy(
    L=1, num_stages=NUM_STAGES, num_epochs=NUM_EPOCHS,
    device=DEVICE, save_dir=CHECKPOINT_DIR, plot_dir=PLOT_DIR,
    num_workers=None, multi_gpu=True, use_g_func=True, resume=False,
    edge_weight=0.0, multi_scale=False,
):
    if multi_scale:
        from models.multi_scale_wrapper import MultiScaleTNRDWrapper
    if num_workers is None:
        num_workers = DATALOADER_NUM_WORKERS

    g_suffix = "" if use_g_func else "_nog"

    if resume:
        start_stage, load_path = _find_resume_stage(L, num_stages, save_dir, g_suffix)
        if start_stage is None:
            print(f"\n  [L={L}] Already complete — skipping.")
            return None, None
        print(f"\n  [L={L}] Resuming from stage {start_stage+1}/{num_stages}")
    else:
        start_stage = 0
        load_path = None

    print(f"\n{'#'*60}")
    print(f"  GREEDY TRAINING  L={L}  T={num_stages}  use_g={use_g_func}")
    if resume:
        print(f"  RESUME from stage {start_stage+1}")
    print(f"  device={device}  workers={num_workers}")
    print(f"  K={K}  epochs={num_epochs}  batch={BATCH_SIZE}  LR={LR}")
    print(f"  RBF: [{RBF_CENTER_MIN}, {RBF_CENTER_MAX}] n={RBF_NUM_CENTERS}")
    print(f"  train: {CLEAN_TRAIN_DIR}")
    print(f"  val  : {CLEAN_VAL_DIR}  (max {VAL_MAX_IMAGES or 'all'} images)")

    if L == 1:
        sched = CURRICULUM_L1_SCHEDULE[:num_stages]
        print(f"\n  CURRICULUM (L=1): stage noise levels = {sched}")
        print(f"  Stages 0-{sched.index(1)-1}: warm-up on easier noise")
        print(f"  Stages {sched.index(1)}-{num_stages-1}: train on target L=1")
    print(f"{'#'*60}")

    model = InertialTNRDNetwork(
        num_stages    = num_stages,
        num_filters   = NUM_FILTERS,
        filter_size   = FILTER_SIZE,
        gamma_inertia = GAMMA_INERTIA,
        sigma_smooth  = SIGMA_SMOOTH,
        nu            = NU,
        K_thresh      = K,
        num_centers   = RBF_NUM_CENTERS,
        use_g_func    = use_g_func,
        device        = device,
    ).to(device)

    if load_path is not None:
        sd = torch.load(load_path, map_location="cpu")
        model.load_state_dict(sd)
        model.to(device)
        print(f"  Loaded checkpoint: {load_path}")
    else:
        model.print_param_summary()

    if multi_scale:
        model = MultiScaleTNRDWrapper(model)
        print(f"  Multi-scale enabled: scales={model.scales}")

    if multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"  DataParallel across {torch.cuda.device_count()} GPUs")

    val_loader_target = make_val_loader(
        CLEAN_VAL_DIR, L=L,
        batch_size=1, patch_size=PATCH_SIZE, seed=42,
        num_workers=num_workers,
        max_images=VAL_MAX_IMAGES if VAL_MAX_IMAGES > 0 else None,
    )

    all_history = {}

    for stage_idx in range(start_stage, num_stages):

        if L == 1:
            train_L = CURRICULUM_L1_SCHEDULE[stage_idx] \
                      if stage_idx < len(CURRICULUM_L1_SCHEDULE) else 1
        else:
            train_L = L

        print(f"\n  Stage {stage_idx+1}: training on L={train_L} "
              f"{'(curriculum warm-up)' if train_L != L else '(target noise)'}")

        train_loader = make_train_loader(
            CLEAN_TRAIN_DIR, L=train_L,
            batch_size=BATCH_SIZE, patch_size=PATCH_SIZE,
            num_workers=num_workers,
        )

        val_loader = make_val_loader(
            CLEAN_VAL_DIR, L=L,
            batch_size=1, patch_size=PATCH_SIZE, seed=42,
            num_workers=num_workers,
            max_images=VAL_MAX_IMAGES if VAL_MAX_IMAGES > 0 else None,
        )

        hist = train_stage(
            model=model, stage_idx=stage_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device, num_epochs=num_epochs,
            lr=LR, L=L,
            use_g_func=use_g_func,
            edge_weight=edge_weight,
            save_dir=save_dir, plot_dir=plot_dir,
        )
        all_history[f"stage_{stage_idx+1}"] = hist

    os.makedirs(save_dir, exist_ok=True)
    hist_path = os.path.join(save_dir, f"training_history_L{L}{g_suffix}.json")
    old_hist = {}
    if resume and os.path.isfile(hist_path):
        with open(hist_path) as fp:
            old_hist = json.load(fp)
    old_hist.update(all_history)
    with open(hist_path, "w") as fp:
        json.dump(old_hist, fp, indent=2)
    print(f"\n  History → {hist_path}")

    final_base = model.module if isinstance(model, nn.DataParallel) else model
    final_path = os.path.join(save_dir, f"model_L{L}{g_suffix}_final.pth")
    final_base.save(final_path)
    print(f"  Final model → {final_path}")

    return model, old_hist



def main():
    parser = argparse.ArgumentParser(description="Train Inertial TNRD Despeckling")
    parser.add_argument("--L",            type=int, default=1)
    parser.add_argument("--stages",       type=int, default=NUM_STAGES)
    parser.add_argument("--epochs",       type=int, default=NUM_EPOCHS)
    parser.add_argument("--both",         action="store_true",
                        help="Train both L=1 and L=10")
    parser.add_argument("--no_g",         action="store_true",
                        help="Train without g_func (ablation A2)")
    parser.add_argument("--workers",      type=int, default=None)
    parser.add_argument("--no-multi-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run/resume all L values in sequence (1, 10)")
    parser.add_argument("--edge-weight", type=float, default=0.0,
                        help="Weight for gradient-magnitude edge loss (default 0)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Enable multi-scale processing (MSND-style)")
    args = parser.parse_args()

    device = DEVICE
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    use_g = not args.no_g

    common = dict(
        device=device, num_workers=args.workers,
        multi_gpu=not args.no_multi_gpu, use_g_func=use_g, resume=args.resume,
        edge_weight=args.edge_weight, multi_scale=args.multi_scale,
    )

    if args.pipeline:
        for L in [1, 10]:
            ns = 10 if L == 1 else min(args.stages, 5)
            train_greedy(L=L, num_stages=ns, num_epochs=args.epochs, **common)
    elif args.both:
        for L in NOISE_LEVELS:
            train_greedy(L=L, num_stages=args.stages, num_epochs=args.epochs, **common)
    else:
        train_greedy(L=args.L, num_stages=args.stages, num_epochs=args.epochs, **common)


if __name__ == "__main__":
    main()