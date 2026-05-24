#!/usr/bin/env python3
"""
download_data.py — fetch standard denoising train/val images into data/

BSD400 / BSD68 names in this repo follow the image-denoising literature. The
easiest stable source is the public DnCNN repository (Zhang et al.):

  • "BSD400" here  ←  Train400  (400 PNGs, ~180×180 crops from BSD + similar)
  • "BSD68" here   ←  Set68     (68 PNGs, common validation set)

This is enough to run train.py / evaluate.py. Full-resolution BSDS500 is
optional (see --bsds500). Set12 (12 images) is used for quick evaluation when
present (see config CLEAN_TEST_DIR).

Usage (from project root):

  python download_data.py
  python download_data.py --force
  python download_data.py --bsds500
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

# ── project paths ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DATA_ROOT  # noqa: E402

# DnCNN (cszn/DnCNN) raw files on GitHub — no git clone needed
_TRAIN400_BASE = (
    "https://raw.githubusercontent.com/cszn/DnCNN/master/"
    "TrainingCodes/DnCNN_TrainingCodes_v1.0/data/Train400/"
)
_SET68_BASE = (
    "https://raw.githubusercontent.com/cszn/DnCNN/master/"
    "TrainingCodes/DnCNN_TrainingCodes_v1.1/data/Test/Set68/"
)
_SET12_BASE = (
    "https://raw.githubusercontent.com/cszn/DnCNN/master/"
    "TrainingCodes/DnCNN_TrainingCodes_v1.1/data/Test/Set12/"
)

# Berkeley Segmentation Benchmark (full BSDS500, ~70 MB)
_BSDS500_TGZ = (
    "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/"
    "grouping/BSR/BSR_bsds500.tgz"
)


def _fetch(url: str, dest: Path, *, force: bool) -> bool:
    """Download url to dest. Returns True if a download was performed."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return False
    req = urllib.request.Request(url, headers={"User-Agent": "MMIP-project/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    return True


def download_train400(out_dir: Path, *, force: bool) -> tuple[int, int]:
    """400 images: test_001.png … test_400.png"""
    n_ok, n_skip = 0, 0
    for i in range(1, 401):
        name = f"test_{i:03d}.png"
        url = _TRAIN400_BASE + name
        dest = out_dir / name
        if _fetch(url, dest, force=force):
            n_ok += 1
        else:
            n_skip += 1
        if i % 50 == 0:
            print(f"  Train400: {i}/400")
    return n_ok, n_skip


def download_set68(out_dir: Path, *, force: bool) -> tuple[int, int]:
    """68 images: test001.png … test068.png"""
    n_ok, n_skip = 0, 0
    for i in range(1, 69):
        name = f"test{i:03d}.png"
        url = _SET68_BASE + name
        dest = out_dir / name
        if _fetch(url, dest, force=force):
            n_ok += 1
        else:
            n_skip += 1
    return n_ok, n_skip


def download_set12(out_dir: Path, *, force: bool) -> tuple[int, int]:
    """12 classic test images: 01.png … 12.png (small / fast evaluation)."""
    n_ok, n_skip = 0, 0
    for i in range(1, 13):
        name = f"{i:02d}.png"
        url = _SET12_BASE + name
        dest = out_dir / name
        if _fetch(url, dest, force=force):
            n_ok += 1
        else:
            n_skip += 1
    return n_ok, n_skip


def download_bsds500_extract_train_val(
    cache_tgz: Path,
    train_out: Path,
    *,
    force: bool,
) -> None:
    """
    Download BSDS500 tarball and copy train/ + val/ JPEGs into train_out.
    Typically ~300 images (not 400); useful if you want full-res BSD patches.
    """
    if not cache_tgz.exists() or force:
        print(f"  Downloading BSDS500 → {cache_tgz} (may take a minute)…")
        _fetch(_BSDS500_TGZ, cache_tgz, force=force)
    else:
        print(f"  Using existing archive {cache_tgz}")

    extract_root = cache_tgz.parent / "_bsds500_extract"
    if not extract_root.exists() or force:
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"  Extracting to {extract_root}…")
        with tarfile.open(cache_tgz, "r:*") as tar:
            tar.extractall(extract_root)

    # Layout after extract: often BSR/BSDS500/data/images/{train,val,test}
    images_root = None
    for p in extract_root.rglob("images"):
        if p.is_dir() and (p / "train").is_dir():
            images_root = p
            break
    if images_root is None:
        raise RuntimeError(
            "Could not find BSDS500 …/images/train inside the archive. "
            "Check extract_root manually: " + str(extract_root)
        )

    train_out.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    n = 0
    for split in ("train", "val"):
        sp = images_root / split
        if not sp.is_dir():
            continue
        for f in sorted(sp.iterdir()):
            if f.suffix.lower() not in exts:
                continue
            dest = train_out / f"{split}_{f.name}"
            if dest.exists() and not force:
                n += 1
                continue
            dest.write_bytes(f.read_bytes())
            n += 1
    print(f"  Copied {n} images from BSDS500 train+val → {train_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Train400 / Set68 / optional BSDS500")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    ap.add_argument(
        "--bsds500",
        action="store_true",
        help="Also download full BSDS500 (~70 MB) into data/_bsds500 and "
        "copy train+val JPEGs to data/BSD400_bsds500 (does not replace BSD400)",
    )
    args = ap.parse_args()

    data = Path(DATA_ROOT)
    train_dir = data / "BSD400"
    val_dir = data / "BSD68"
    set12_dir = data / "Set12"

    print("DnCNN Train400 →", train_dir)
    a, b = download_train400(train_dir, force=args.force)
    print(f"  Done. Downloaded: {a}, skipped (already there): {b}")

    print("DnCNN Set68 →", val_dir)
    a, b = download_set68(val_dir, force=args.force)
    print(f"  Done. Downloaded: {a}, skipped (already there): {b}")

    print("DnCNN Set12 (small test) →", set12_dir)
    a, b = download_set12(set12_dir, force=args.force)
    print(f"  Done. Downloaded: {a}, skipped (already there): {b}")

    if args.bsds500:
        data.mkdir(parents=True, exist_ok=True)
        alt_train = data / "BSD400_bsds500"
        print("Optional BSDS500 train+val →", alt_train)
        download_bsds500_extract_train_val(
            data / "BSR_bsds500.tgz",
            alt_train,
            force=args.force,
        )
        print("  Training auto-uses this folder when USE_LARGE_TRAIN is True in config.py.")

    print("\nNext: python train.py  |  python evaluate.py")
    print("Note: With Set12 present, evaluate.py uses 12 images; else data/test_images. SAR: add manually.")


if __name__ == "__main__":
    main()
