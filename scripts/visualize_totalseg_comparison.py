#!/usr/bin/env python3
"""
Visual comparison between real and generated CT volumes using TotalSegmentator overlays.

For each sampled patient, the script shows axial/coronal/sagittal views with:
  - real CT + real TotalSegmentator contours
  - generated CT + generated TotalSegmentator contours
  - agreement map (real-only / gen-only / overlap) for selected organs

Default organs focus on prompt-relevant anatomy used in prompt_generation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap

DEFAULT_ORGANS = [
    "liver", "spleen", "pancreas", "stomach",
    "duodenum", "small_bowel", "colon",
    "kidney_left", "kidney_right", "urinary_bladder",
    "uterus", "rectum",
]

ORGAN_COLORS = {
    "liver": "#ef476f",
    "spleen": "#f78c6b",
    "pancreas": "#ffd166",
    "stomach": "#06d6a0",
    "duodenum": "#00b4d8",
    "small_bowel": "#118ab2",
    "colon": "#073b4c",
    "kidney_left": "#8e7dff",
    "kidney_right": "#5e60ce",
    "urinary_bladder": "#4cc9f0",
    "uterus": "#ff99c8",
    "rectum": "#9d4edd",
}

AGREEMENT_CMAP = ListedColormap([
    "#000000",  # background
    "#ef476f",  # real only
    "#118ab2",  # generated only
    "#06d6a0",  # overlap
])


def load_split_paths(test_json: str, split: str) -> list[str]:
    with open(test_json) as f:
        data = json.load(f)
    key_map = {
        "train": "training", "training": "training",
        "val": "validation", "validation": "validation",
        "test": "test", "testing": "test",
    }
    key = key_map.get(split, split)
    return [item["image"] for item in data[key]]


def stratified_sample(paths: list[str], n: int, seed: int = 42) -> list[str]:
    by_year: dict[str, list[str]] = defaultdict(list)
    for p in paths:
        for part in p.split("/"):
            if part.isdigit() and len(part) == 4:
                by_year[part].append(p)
                break
        else:
            by_year["unknown"].append(p)

    rng = random.Random(seed)
    years = sorted(by_year)
    for year in years:
        rng.shuffle(by_year[year])

    selected: list[str] = []
    idx = {year: 0 for year in years}
    while len(selected) < n:
        added = False
        for year in years:
            if idx[year] < len(by_year[year]) and len(selected) < n:
                selected.append(by_year[year][idx[year]])
                idx[year] += 1
                added = True
        if not added:
            break
    return selected


def strip_data_prefix(image_path: str) -> str:
    tail = image_path
    for prefix in ("./data/private_ct/", "data/private_ct/", "./data/private_ct", "data/private_ct"):
        if tail.startswith(prefix):
            tail = tail[len(prefix):]
            break
    return tail.lstrip("/")


def resolve_real_gen_paths(image_path: str, real_root: str, gen_root: str) -> tuple[str, str, str]:
    rel = strip_data_prefix(image_path)
    patient_id = Path(rel).parent.name
    return os.path.join(real_root, rel), os.path.join(gen_root, rel), patient_id


def resolve_generated_root(gen_root: str, checkpoint_tag: str | None) -> str:
    root = Path(gen_root)
    if checkpoint_tag:
        return str(root / checkpoint_tag)
    return str(root)


def load_ct(path: str) -> np.ndarray:
    proxy = nib.load(path)
    vol = np.array(proxy.dataobj, dtype=np.float32)
    proxy.uncache()
    return vol


def normalize_ct(vol: np.ndarray) -> np.ndarray:
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax <= vmin:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)


def load_multilabel_seg(seg_path: str) -> tuple[np.ndarray, dict]:
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti

    seg_img, label_map = load_multilabel_nifti(seg_path)
    seg = seg_img.get_fdata(dtype=np.float32).astype(np.int32)
    return seg, label_map


def build_selected_label_ids(label_map: dict, organ_names: list[str]) -> list[tuple[str, int]]:
    name_to_id = {v: int(k) for k, v in label_map.items()}
    selected = []
    for organ in organ_names:
        if organ in name_to_id:
            selected.append((organ, name_to_id[organ]))
    return selected


def build_union_mask(seg: np.ndarray, selected_ids: list[tuple[str, int]]) -> np.ndarray:
    if not selected_ids:
        return np.zeros_like(seg, dtype=bool)
    ids = np.array([label_id for _, label_id in selected_ids], dtype=np.int32)
    return np.isin(seg, ids)


def best_slice_indices(reference_mask: np.ndarray) -> dict[str, int]:
    if not reference_mask.any():
        h, w, d = reference_mask.shape
        return {"axial": d // 2, "coronal": w // 2, "sagittal": h // 2}

    axial_scores = reference_mask.sum(axis=(0, 1))
    coronal_scores = reference_mask.sum(axis=(0, 2))
    sagittal_scores = reference_mask.sum(axis=(1, 2))
    return {
        "axial": int(np.argmax(axial_scores)),
        "coronal": int(np.argmax(coronal_scores)),
        "sagittal": int(np.argmax(sagittal_scores)),
    }


def slice2d(vol: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "axial":
        return vol[:, :, index]
    if axis == "coronal":
        return vol[:, index, :]
    return vol[index, :, :]


def add_contours(ax, seg2d: np.ndarray, selected_ids: list[tuple[str, int]]):
    for organ_name, label_id in selected_ids:
        mask = seg2d == label_id
        if mask.any():
            ax.contour(mask.T.astype(np.float32), levels=[0.5], colors=[ORGAN_COLORS.get(organ_name, "yellow")], linewidths=0.9)


def agreement_map(real_seg2d: np.ndarray, gen_seg2d: np.ndarray, selected_ids: list[tuple[str, int]]) -> np.ndarray:
    ids = np.array([label_id for _, label_id in selected_ids], dtype=np.int32)
    real_mask = np.isin(real_seg2d, ids)
    gen_mask = np.isin(gen_seg2d, ids)

    out = np.zeros(real_seg2d.shape, dtype=np.uint8)
    out[real_mask & ~gen_mask] = 1
    out[~real_mask & gen_mask] = 2
    out[real_mask & gen_mask] = 3
    return out


def organ_volume_table(real_seg: np.ndarray, gen_seg: np.ndarray, selected_ids: list[tuple[str, int]]) -> str:
    rows = []
    for organ_name, label_id in selected_ids:
        real_vox = int((real_seg == label_id).sum())
        gen_vox = int((gen_seg == label_id).sum())
        if real_vox == 0 and gen_vox == 0:
            continue
        delta = gen_vox - real_vox
        rows.append(f"{organ_name}: R={real_vox} G={gen_vox} Δ={delta:+d}")
    return " | ".join(rows[:5]) if rows else "No selected organs found"


def plot_samples(samples: list[dict], out_path: Path, organ_names: list[str]) -> None:
    n = len(samples)
    fig, axes = plt.subplots(
        nrows=n * 3,
        ncols=3,
        figsize=(14, 4 * n),
        gridspec_kw={"wspace": 0.03, "hspace": 0.18},
    )
    if n == 1:
        axes = axes.reshape(3, 3)

    view_names = ["axial", "coronal", "sagittal"]
    titles = ["Real + TotalSeg", "Generated + TotalSeg", "Agreement map"]

    for sample_idx, sample in enumerate(samples):
        for view_idx, view in enumerate(view_names):
            row = sample_idx * 3 + view_idx
            ax_real = axes[row, 0] if n > 1 else axes[view_idx, 0]
            ax_gen = axes[row, 1] if n > 1 else axes[view_idx, 1]
            ax_agree = axes[row, 2] if n > 1 else axes[view_idx, 2]

            slice_idx = sample["slice_idx"][view]
            real_ct = slice2d(sample["real_ct"], view, slice_idx)
            gen_ct = slice2d(sample["gen_ct"], view, slice_idx)
            real_seg = slice2d(sample["real_seg"], view, slice_idx)
            gen_seg = slice2d(sample["gen_seg"], view, slice_idx)
            agree = agreement_map(real_seg, gen_seg, sample["selected_ids"])

            ax_real.imshow(real_ct.T, cmap="gray", vmin=0, vmax=1, origin="lower")
            ax_gen.imshow(gen_ct.T, cmap="gray", vmin=0, vmax=1, origin="lower")
            ax_agree.imshow(agree.T, cmap=AGREEMENT_CMAP, vmin=0, vmax=3, origin="lower")

            add_contours(ax_real, real_seg, sample["selected_ids"])
            add_contours(ax_gen, gen_seg, sample["selected_ids"])

            for ax in (ax_real, ax_gen, ax_agree):
                ax.set_xticks([])
                ax.set_yticks([])

            if view_idx == 0:
                for col_idx, title in enumerate(titles):
                    ax = axes[row, col_idx] if n > 1 else axes[view_idx, col_idx]
                    ax.set_title(title, fontsize=11, fontweight="bold")

            ax_real.set_ylabel(f"{view}\nidx={slice_idx}", fontsize=9)

    legend_lines = [
        "Agreement: red=real only | blue=generated only | green=overlap",
        "Organs: " + ", ".join(organ_names),
    ]
    fig.suptitle("Real vs Generated TotalSegmentator comparison", fontsize=14, y=0.995)
    fig.text(0.5, 0.005, "\n".join(legend_lines), ha="center", va="bottom", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved visualization to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize real vs generated CT with TotalSegmentator overlays")
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--real_seg_root", required=True)
    ap.add_argument("--gen_seg_root", required=True)
    ap.add_argument("--test_json", default="dataset/unet_test_data_volumes.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint_tag", default=None)
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    organ_names = [x.strip() for x in args.organs.split(",") if x.strip()]
    gen_root = resolve_generated_root(args.gen_root, args.checkpoint_tag)
    gen_seg_root = Path(args.gen_seg_root)
    if args.checkpoint_tag:
        gen_seg_root = gen_seg_root / args.checkpoint_tag

    paths = load_split_paths(args.test_json, args.split)
    selected_paths = stratified_sample(paths, args.n_samples, args.seed)

    samples = []
    for image_path in selected_paths:
        real_path, gen_path, patient_id = resolve_real_gen_paths(image_path, args.real_root, gen_root)
        real_seg_path = Path(args.real_seg_root) / patient_id / "totalseg.nii.gz"
        gen_seg_path = gen_seg_root / patient_id / "totalseg.nii.gz"

        if not (os.path.exists(real_path) and os.path.exists(gen_path) and real_seg_path.exists() and gen_seg_path.exists()):
            print(f"SKIP {patient_id}: missing CT or segmentation file")
            continue

        real_ct = normalize_ct(load_ct(real_path))
        gen_ct = normalize_ct(load_ct(gen_path))
        real_seg, label_map = load_multilabel_seg(str(real_seg_path))
        gen_seg, _ = load_multilabel_seg(str(gen_seg_path))

        selected_ids = build_selected_label_ids(label_map, organ_names)
        real_mask = build_union_mask(real_seg, selected_ids)
        slice_idx = best_slice_indices(real_mask)
        summary = organ_volume_table(real_seg, gen_seg, selected_ids)

        samples.append({
            "patient_id": patient_id,
            "real_ct": real_ct,
            "gen_ct": gen_ct,
            "real_seg": real_seg,
            "gen_seg": gen_seg,
            "selected_ids": selected_ids,
            "slice_idx": slice_idx,
            "summary": summary,
        })

    if not samples:
        raise SystemExit("No valid samples found. Check whether TotalSegmentator outputs are ready.")

    plot_samples(samples, Path(args.output), organ_names)


if __name__ == "__main__":
    main()
