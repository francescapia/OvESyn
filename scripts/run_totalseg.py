#!/usr/bin/env python3
"""
run_totalseg.py
Run TotalSegmentator on real (test set) and/or generated volumes.

Volumes are stored in [0,1] normalized range (originally [-1000, 1000] HU).
This script rescales them back to HU before passing to TotalSegmentator,
then discards the temp file.

Output structure:
  <out_dir>/real/<patient_id>/totalseg.nii.gz
  <out_dir>/generated/<checkpoint_tag>/<patient_id>/totalseg.nii.gz

Usage:
  python scripts/run_totalseg.py \
      --val_json dataset/unet_test_data_volumes.json \
      --split test \
      --data_base_dir ./data/private_ct_preprocessed \
      --gen_root "./dataset/eval_generations/unet_clipFT_FT" \
      --checkpoint_tag unet_clipFT_FT_50 \
      --out_dir ./results/totalseg \
      --mode both
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_totalseg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json_paths(val_json: str, split: str) -> List[str]:
    with open(val_json) as f:
        j = json.load(f)
    key_map = {
        "train": "training", "training": "training",
        "val": "validation", "validation": "validation",
        "test": "test", "testing": "test",
    }
    key = key_map.get(split, split)
    if key not in j:
        raise KeyError(f"Split '{split}' not found. Available: {list(j.keys())}")
    return [x["image"] for x in j[key]]


def patient_id_from_path(json_path: str) -> str:
    """Extract patient folder name (e.g. CASE_SYN_0001) from JSON path."""
    return Path(json_path).parent.name


def resolve_real_path(json_path: str, data_base_dir: str) -> str:
    tail = json_path
    for prefix in ("./data/private_ct/", "data/private_ct/", "./data/private_ct", "data/private_ct"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
            break
    return os.path.join(data_base_dir, tail.lstrip("/"))


def resolve_gen_path(json_path: str, gen_root: str, checkpoint_tag: Optional[str]) -> str:
    tail = json_path
    for prefix in ("./data/private_ct/", "data/private_ct/", "./data/private_ct", "data/private_ct"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
            break
    base = Path(gen_root)
    if checkpoint_tag:
        return str(base / checkpoint_tag / tail.lstrip("/"))
    return str(base / tail.lstrip("/"))


def rescale_to_hu(data: np.ndarray, hu_min: float = -1000.0, hu_max: float = 1000.0) -> np.ndarray:
    """Rescale [0, 1] normalized volume back to HU range."""
    return (data * (hu_max - hu_min) + hu_min).astype(np.float32)


def run_totalseg(input_nii: nib.Nifti1Image, output_path: Path, fast: bool, device: str) -> bool:
    """
    Rescale to HU, write a temp NIfTI, run TotalSegmentator, move output.
    Returns True on success, False on failure.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        raise SystemExit(
            "TotalSegmentator not installed. Run:\n"
            "  pip install totalsegmentator"
        )

    seg_out = output_path / "totalseg.nii.gz"

    if seg_out.exists():
        logger.info(f"  [SKIP] already exists: {seg_out}")
        return True

    # Rescale to HU in a temp file
    data_norm = input_nii.get_fdata(dtype=np.float32)
    data_hu = rescale_to_hu(data_norm)
    hu_nii = nib.Nifti1Image(data_hu, input_nii.affine, input_nii.header)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "input_hu.nii.gz"
        # In ml mode, TotalSegmentator expects a file output path.
        tmp_out_file = Path(tmpdir) / "segmentation_ml.nii.gz"
        tmp_out_dir = Path(tmpdir) / "seg_out_dir"
        tmp_out_dir.mkdir()

        nib.save(hu_nii, str(tmp_in))

        try:
            totalsegmentator(
                input=tmp_in,
                output=tmp_out_file,
                ml=True,          # single multilabel NIfTI
                fast=fast,
                device=device,
                quiet=True,
            )
        except Exception as e:
            logger.error(f"  TotalSegmentator failed: {e}")
            return False

        seg_file = None
        # Preferred: exact file path passed to API.
        if tmp_out_file.exists():
            seg_file = tmp_out_file
        else:
            # Fallbacks for potential API/path behavior differences.
            candidates = list(Path(tmpdir).glob("*.nii.gz")) + list(tmp_out_dir.glob("*.nii.gz"))
            if candidates:
                seg_file = candidates[0]

        if seg_file is None:
            logger.error(f"  No segmentation output found in temporary folder: {tmpdir}")
            return False

        import shutil
        output_path.mkdir(parents=True, exist_ok=True)
        shutil.move(str(seg_file), str(seg_out))

    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run TotalSegmentator on real + generated volumes")
    ap.add_argument("--val_json", required=True, help="JSON with data splits")
    ap.add_argument("--split", required=True, choices=["train", "training", "val", "validation", "test"])
    ap.add_argument("--data_base_dir", required=True, help="Root of preprocessed real data")
    ap.add_argument("--gen_root", default=None, help="Root of generated volumes")
    ap.add_argument("--checkpoint_tag", default=None, help="Checkpoint subfolder inside gen_root")
    ap.add_argument("--out_dir", required=True, help="Where to save totalseg outputs")
    ap.add_argument("--mode", choices=["real", "generated", "both"], default="both")
    ap.add_argument("--fast", action="store_true", help="Use TotalSegmentator fast mode (lower GPU memory)")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    paths = load_json_paths(args.val_json, args.split)
    logger.info(f"Split '{args.split}': {len(paths)} volumes | mode={args.mode}")

    do_real = args.mode in ("real", "both")
    do_gen = args.mode in ("generated", "both")

    if do_gen and args.gen_root is None:
        raise SystemExit("--gen_root is required when mode includes 'generated'")

    # ------------------------------------------------------------------
    # REAL volumes
    # ------------------------------------------------------------------
    if do_real:
        logger.info("=" * 55)
        logger.info("Processing REAL volumes")
        logger.info("=" * 55)
        real_out_base = out_dir / "real"
        
        # Check if real segmentations already exist
        existing_real_segs = list(real_out_base.glob("*/totalseg.nii.gz"))
        if len(existing_real_segs) >= len(paths):
            logger.info(
                f"Real segmentations already complete ({len(existing_real_segs)} files found). "
                f"Skipping real processing."
            )
        else:
            ok = fail = skip = 0
            for jp in tqdm(paths, desc="Real"):
                pid = patient_id_from_path(jp)
                vol_path = resolve_real_path(jp, args.data_base_dir)
                if not os.path.exists(vol_path):
                    logger.warning(f"Missing real volume: {vol_path}")
                    fail += 1
                    continue

                seg_out = real_out_base / pid
                if (seg_out / "totalseg.nii.gz").exists():
                    skip += 1
                    continue

                nii = nib.load(vol_path)
                success = run_totalseg(nii, seg_out, fast=args.fast, device=args.device)
                if success:
                    ok += 1
                else:
                    fail += 1

            logger.info(f"Real — done: {ok}, skip: {skip}, fail: {fail}")

    # ------------------------------------------------------------------
    # GENERATED volumes
    # ------------------------------------------------------------------
    if do_gen:
        logger.info("=" * 55)
        logger.info("Processing GENERATED volumes")
        logger.info("=" * 55)
        ckpt = args.checkpoint_tag or "unknown"
        gen_out_base = out_dir / "generated" / ckpt
        ok = fail = skip = 0
        for jp in tqdm(paths, desc=f"Generated ({ckpt})"):
            pid = patient_id_from_path(jp)
            vol_path = resolve_gen_path(jp, args.gen_root, args.checkpoint_tag)
            if not os.path.exists(vol_path):
                logger.warning(f"Missing generated volume: {vol_path}")
                fail += 1
                continue

            seg_out = gen_out_base / pid
            if (seg_out / "totalseg.nii.gz").exists():
                skip += 1
                continue

            nii = nib.load(vol_path)
            success = run_totalseg(nii, seg_out, fast=args.fast, device=args.device)
            if success:
                ok += 1
            else:
                fail += 1

        logger.info(f"Generated — done: {ok}, skip: {skip}, fail: {fail}")

    logger.info("All done.")


if __name__ == "__main__":
    main()
