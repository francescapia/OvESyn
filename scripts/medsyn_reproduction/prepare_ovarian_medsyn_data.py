#!/usr/bin/env python3
"""Prepare ovarian CT volumes and prompt text for MedSyn low-resolution stage."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_paths(json_path: Path, split: str) -> list[str]:
    with json_path.open() as f:
        data = json.load(f)
    aliases = {
        "train": "training",
        "training": "training",
        "val": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }
    key = aliases.get(split, split)
    if key not in data:
        raise KeyError(f"Split {split!r} not found in {json_path}; available={list(data)}")
    return [item["image"] for item in data[key]]


def patient_id(path: str) -> str:
    match = re.search(r"\bIEO\d+\b", str(path))
    if match:
        return match.group(0)
    return Path(path).parent.name


def resolve_real_path(path: str, data_base_dir: Path) -> Path:
    tail = str(path)
    for prefix in ("./data/private_ct/", "data/private_ct/", "./data/private_ct", "data/private_ct"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
            break
    return data_base_dir / tail.lstrip("/")


def load_reports(reports_csv: Path) -> dict[str, str]:
    df = pd.read_csv(reports_csv).fillna("")
    if "VolumeName" not in df.columns:
        raise ValueError(f"Expected VolumeName in {reports_csv}")
    text_cols = [c for c in ("Findings_EN", "Impressions_EN", "Findings", "Impressions") if c in df.columns]
    if not text_cols:
        raise ValueError(f"No report text columns found in {reports_csv}")

    out: dict[str, str] = {}
    for _, row in df.iterrows():
        text = " ".join(str(row[c]).strip() for c in text_cols if str(row[c]).strip())
        out[str(row["VolumeName"])] = re.sub(r"\s+", " ", text).strip()
    return out


def load_volume(path: Path) -> np.ndarray:
    try:
        from monai.transforms import LoadImage
    except Exception as e:
        raise SystemExit("MONAI is required to prepare MedSyn data on the cluster.") from e

    loader = LoadImage(image_only=True)
    arr = loader(str(path))
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 4:
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume for {path}, got shape {arr.shape}")
    return arr


def normalize_to_unit(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not finite.all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if arr.min() < -0.01 or arr.max() > 1.01:
        arr = np.clip(arr, -1000.0, 1000.0)
        arr = (arr + 1000.0) / 2000.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def make_medsyn_array(arr: np.ndarray, size: int) -> np.ndarray:
    # MedSyn expects [C, F, H, W]. Use axial frames [D, H, W].
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr)[None, None]
    x = F.interpolate(x, size=(size, size, size), mode="trilinear", align_corners=False)
    ct = x[0, 0].clamp(0, 1).numpy().astype(np.float32)
    zeros = np.zeros_like(ct, dtype=np.float32)
    return np.stack([ct, zeros, zeros, zeros], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--reports_csv", type=Path, required=True)
    ap.add_argument("--data_base_dir", type=Path, default=Path("./data/private_ct"))
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    image_dir = args.out_root / args.split / "lowres_images"
    prompt_dir = args.out_root / args.split / "prompts_text"
    image_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(args.reports_csv)
    paths = load_paths(args.json, args.split)
    if args.limit > 0:
        paths = paths[: args.limit]

    rows = []
    for idx, raw_path in enumerate(paths):
        pid = patient_id(raw_path)
        real_path = resolve_real_path(raw_path, args.data_base_dir)
        if pid not in reports:
            raise KeyError(f"No report found for {pid} in {args.reports_csv}")
        if not real_path.exists():
            raise FileNotFoundError(real_path)

        arr = normalize_to_unit(load_volume(real_path))
        medsyn_arr = make_medsyn_array(arr, args.size)
        np.save(image_dir / f"{pid}.npy", medsyn_arr)
        (prompt_dir / f"{pid}.txt").write_text(reports[pid] + "\n")
        rows.append({"idx": idx, "volume_name": pid, "real_path": str(real_path), "prompt": reports[pid]})
        print(f"[{idx + 1}/{len(paths)}] wrote {pid}")

    pd.DataFrame(rows).to_csv(args.out_root / args.split / "manifest.csv", index=False)
    print(f"Wrote {len(rows)} cases under {args.out_root / args.split}")


if __name__ == "__main__":
    main()
