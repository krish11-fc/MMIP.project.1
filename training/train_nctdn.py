import os, sys, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config import (
    DEVICE, NUM_STAGES, BATCH_SIZE, PATCH_SIZE,
    NUM_EPOCHS, LR, LR_STEP, LR_GAMMA_SCHED, WEIGHT_DECAY, GRAD_CLIP,
    CLEAN_TRAIN_DIR, CLEAN_VAL_DIR,
    CHECKPOINT_DIR, PLOT_DIR, COMPARISON_DIR,
    GAMMA_INERTIA, SIGMA_SMOOTH, NU, K,
    NUM_FILTERS, FILTER_SIZE, RBF_NUM_CENTERS,
    EMBED_DIM, NUM_NOISE_LEVELS,
    NOISE_LEVELS_ALL,
    DATALOADER_NUM_WORKERS, VAL_MAX_IMAGES, WARMUP_EPOCHS,
)
from models.noise_conditional_network import NoiseConditionalTNRDNetwork
from dataset import make_mixed_train_loader, make_mixed_val_loader
from utils.visualization import plot_loss_curve, plot_psnr_curve


def _psnr255(pred, target):
    with torch.no_grad():
        mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def _ssim255(pred, target):
    p = pred.detach().cpu().float().numpy()
    t = target.detach().cpu().float().numpy()
    if p.ndim == 4:
        scores = [sk_ssim(p[i,0], t[i,0], data_range=255.0) for i in range(p.shape[0])]
        return float(np.mean(scores))
    return float(sk_ssim(p.squeeze(), t.squeeze(), data_range=255.0))


def train_one_epoch(model, loader, optimizer, device, active_stages):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for u_gt, f, L_batch in loader:
        u_gt, f = u_gt.to(device), f.to(device)
        L_batch = L_batch.to(device)
        optimizer.zero_grad()
        u_pred, _ = model(f, L=L_batch, active_stages=active_stages)
        loss = criterion(u_pred, u_gt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, device, active_stages, L):
    model.eval()
    psnr_list, ssim_list = [], []
    for u_gt, f in loader:
        u_gt = u_gt.to(device)
        f = f.to(device)
        u_pred, _ = model(f, L=L, active_stages=active_stages)
        u_pred = u_pred.clamp(0.0, 255.0)
        psnr_list.append(_psnr255(u_pred, u_gt))
        ssim_list.append(_ssim255(u_pred, u_gt))
    return float(np.mean(psnr_list)), float(np.mean(ssim_list))


def _set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def train_stage(model, stage_idx, train_loader, val_loaders, device,
                num_epochs=NUM_EPOCHS, lr=LR, save_dir=CHECKPOINT_DIR,
                plot_dir=PLOT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    base = model.module if isinstance(model, nn.DataParallel) else model
    base.freeze_stages(up_to=stage_idx)
    base.unfreeze_stage(stage_idx)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)

    print(f"\n{'='*60}")
    print(f"  NCTDN Mixed — Stage {stage_idx+1}/{base.T}")
    print(f"  trainable params: {n_trainable:,}")
    print(f"  epochs={num_epochs}  lr={lr}")
    print(f"{'='*60}")

    optimizer = optim.Adam(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA_SCHED)
    active = stage_idx + 1

    history = {"train_loss": [], "val_psnr": {}}
    best_avg = -1.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        if epoch <= WARMUP_EPOCHS:
            _set_lr(optimizer, lr * epoch / WARMUP_EPOCHS)
        elif epoch == WARMUP_EPOCHS + 1:
            _set_lr(optimizer, lr)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, active)
        history["train_loss"].append(train_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            val_psnrs = {}
            for val_L, val_loader in val_loaders.items():
                vp, vs = validate(model, val_loader, device, active, val_L)
                val_psnrs[f"L{val_L}"] = vp
            avg_psnr = float(np.mean(list(val_psnrs.values())))
            history["val_psnr"][epoch] = val_psnrs

            if epoch > WARMUP_EPOCHS:
                scheduler.step()

            cur_lr = optimizer.param_groups[0]["lr"]
            psnr_str = "  ".join([f"L{k[1:]}={v:.2f}" for k, v in val_psnrs.items()])
            print(f"  Ep [{epoch:3d}/{num_epochs}]  loss={train_loss:.2f}  "
                  f"{psnr_str}  avg={avg_psnr:.2f}  lr={cur_lr:.2e}  ({time.time()-t0:.1f}s)")

            if avg_psnr > best_avg:
                best_avg = avg_psnr
                base.save(os.path.join(save_dir, f"nctdn_stage{stage_idx+1}_best.pth"))
        elif epoch > WARMUP_EPOCHS:
            scheduler.step()

    base.save(os.path.join(save_dir, f"nctdn_stage{stage_idx+1}_last.pth"))
    return history


def train_mixed(num_stages=NUM_STAGES, num_epochs=NUM_EPOCHS,
                device=DEVICE, save_dir=CHECKPOINT_DIR, plot_dir=PLOT_DIR,
                num_workers=None, multi_gpu=True):
    if num_workers is None:
        num_workers = DATALOADER_NUM_WORKERS

    print(f"\n{'#'*60}")
    print(f"  NCTDN MIXED-NOISE TRAINING — single model for all L")
    print(f"  Noise levels: {NOISE_LEVELS_ALL}")
    print(f"  T={num_stages}  epochs={num_epochs}")
    print(f"{'#'*60}")

    model = NoiseConditionalTNRDNetwork(
        num_stages=num_stages, num_filters=NUM_FILTERS,
        filter_size=FILTER_SIZE, gamma_inertia=GAMMA_INERTIA,
        sigma_smooth=SIGMA_SMOOTH, nu=NU, K_thresh=K,
        num_centers=RBF_NUM_CENTERS, use_g_func=True,
        embed_dim=EMBED_DIM, num_noise_levels=NUM_NOISE_LEVELS,
        device=device,
    ).to(device)
    model.print_param_summary()

    if multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    train_loader = make_mixed_train_loader(
        CLEAN_TRAIN_DIR, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE,
        num_workers=num_workers)

    val_loaders = {}
    for L in NOISE_LEVELS_ALL:
        from dataset import make_val_loader
        val_loaders[L] = make_val_loader(
            CLEAN_VAL_DIR, L=L, batch_size=1, patch_size=PATCH_SIZE,
            seed=42, num_workers=num_workers,
            max_images=VAL_MAX_IMAGES if VAL_MAX_IMAGES > 0 else None)

    all_history = {}
    for stage_idx in range(num_stages):
        hist = train_stage(
            model=model, stage_idx=stage_idx,
            train_loader=train_loader, val_loaders=val_loaders,
            device=device, num_epochs=num_epochs, lr=LR,
            save_dir=save_dir, plot_dir=plot_dir)
        all_history[f"stage_{stage_idx+1}"] = hist

    base = model.module if isinstance(model, nn.DataParallel) else model
    final_path = os.path.join(save_dir, "nctdn_model_mixed_final.pth")
    base.save(final_path)
    print(f"\n  Final model -> {final_path}")

    hist_path = os.path.join(save_dir, "nctdn_training_history_mixed.json")
    with open(hist_path, "w") as fp:
        json.dump(all_history, fp, indent=2)
    print(f"  History -> {hist_path}")

    return model, all_history


def main():
    parser = argparse.ArgumentParser(description="Train NCTDN mixed-noise")
    parser.add_argument("--stages", type=int, default=NUM_STAGES)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--no-multi-gpu", action="store_true")
    args = parser.parse_args()

    device = DEVICE
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    train_mixed(num_stages=args.stages, num_epochs=args.epochs,
                device=device, num_workers=args.workers,
                multi_gpu=not args.no_multi_gpu)


if __name__ == "__main__":
    main()
