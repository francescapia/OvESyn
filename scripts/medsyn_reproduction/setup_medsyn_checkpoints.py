#!/usr/bin/env python3
"""Create Accelerate-style checkpoint folders for MedSyn released weights."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src, dst.parent)
        os.symlink(rel, dst)


def setup_stage(src: Path, out_dir: Path, copy: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    ckpt = out_dir / "0_ckpt"
    ckpt.mkdir(parents=True, exist_ok=True)

    # Accelerate may look for either name depending on how many models were
    # registered. MedSyn samples from ema_model, commonly saved as model_1.
    link_or_copy(src.resolve(), ckpt / "pytorch_model.bin", copy)
    link_or_copy(src.resolve(), ckpt / "pytorch_model_1.bin", copy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1_bin", type=Path, required=True)
    ap.add_argument("--stage2_bin", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--copy", action="store_true", help="Copy instead of symlinking large .bin files.")
    args = ap.parse_args()

    setup_stage(args.stage1_bin, args.out_root / "stage1_pretrained", args.copy)
    setup_stage(args.stage2_bin, args.out_root / "stage2_pretrained", args.copy)
    print(f"Wrote MedSyn checkpoint wrappers under {args.out_root}")


if __name__ == "__main__":
    main()
