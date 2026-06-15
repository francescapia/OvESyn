#!/usr/bin/env python3
"""
Visual comparison between real and generated CT volumes using TotalSegmentator overlays.
One PNG per patient.

Layout:
  rows = axial / coronal / sagittal
  cols = Real + TotalSeg | Generated + TotalSeg | Agreement map

Notes:
  - Assumes CT volumes are already normalized in [0, 1].
  - Reorients NIfTI files to canonical orientation before slicing.
  - Uses spacing-aware aspect ratios for coronal/sagittal views.
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
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba

DEFAULT_ORGANS = [
    "liver", "spleen", "pancreas", "stomach",
    "duodenum", "small_bowel", "colon",
    "kidney_left", "kidney_right", "urinary_bladder",
    "uterus", "rectum",
]

ORGAN_COLORS = {
    "liver":           "#ef476f",
    "spleen":          "#f78c6b",
    "pancreas":        "#ffd166",
    "stomach":         "#06d6a0",
    "duodenum":        "#00b4d8",
    "small_bowel":     "#118ab2",
    "colon":           "#073b4c",
    "kidney_left":     "#8e7dff",
    "kidney_right":    "#5e60ce",
    "urinary_bladder": "#4cc9f0",
    "uterus":          "#ff99c8",
    "rectum":          "#9d4edd",
}

AGREEMENT_CMAP = ListedColormap([
    "#111111",
    "#ef476f",
    "#118ab2",
    "#06d6a0",
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


def load_nifti_canonical(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return vol, zooms


def clamp_ct_01(vol: np.ndarray) -> np.ndarray:
    return np.clip(vol, 0.0, 1.0)


def load_multilabel_seg_canonical(seg_path: str) -> tuple[np.ndarray, dict, tuple[float, float, float]]:
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti

    seg_img, label_map = load_multilabel_nifti(seg_path)
    seg_img = nib.as_closest_canonical(seg_img)
    seg = seg_img.get_fdata(dtype=np.float32).astype(np.int32)
    zooms = tuple(float(z) for z in seg_img.header.get_zooms()[:3])
    return seg, label_map, zooms


def build_selected_label_ids(label_map: dict, organ_names: list[str]) -> list[tuple[str, int]]:
    name_to_id = {v: int(k) for k, v in label_map.items()}
    return [(organ, name_to_id[organ]) for organ in organ_names if organ in name_to_id]


def build_union_mask(seg: np.ndarray, selected_ids: list[tuple[str, int]]) -> np.ndarray:
    if not selected_ids:
        return np.zeros_like(seg, dtype=bool)
    ids = np.array([label_id for _, label_id in selected_ids], dtype=np.int32)
    return np.isin(seg, ids)


def best_slice_indices(reference_mask: np.ndarray) -> dict[str, int]:
    if not reference_mask.any():
        nx, ny, nz = reference_mask.shape
        return {"axial": nz // 2, "coronal": ny // 2, "sagittal": nx // 2}
    return {
        "axial": int(np.argmax(reference_mask.sum(axis=(0, 1)))),
        "coronal": int(np.argmax(reference_mask.sum(axis=(0, 2)))),
        "sagittal": int(np.argmax(reference_mask.sum(axis=(1, 2)))),
    }


def slice2d(vol: np.ndarray, view: str, index: int) -> np.ndarray:
    if view == "axial":
        return vol[:, :, index]
    if view == "coronal":
        return vol[:, index, :]
    if view == "sagittal":
        return vol[index, :, :]
    raise ValueError(f"Unknown view: {view}")


def display2d(arr2d: np.ndarray) -> np.ndarray:
    # Rotate into a conventional display orientation while keeping
    # CT, segmentations, and agreement maps perfectly aligned.
    return np.rot90(arr2d, k=1)


def get_aspect(view: str, zooms: tuple[float, float, float]) -> float:
    x, y, z = zooms
    if view == "axial":
        return y / x
    if view == "coronal":
        return z / x
    if view == "sagittal":
        return z / y
    raise ValueError(f"Unknown view: {view}")


def draw_organ_overlays(
    ax,
    seg2d: np.ndarray,
    selected_ids: list[tuple[str, int]],
    aspect: float,
    alpha: float = 0.45,
) -> None:
    seg_disp = display2d(seg2d)
    h, w = seg_disp.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for organ_name, label_id in selected_ids:
        mask = seg_disp == label_id
        if not mask.any():
            continue
        r, g, b, _ = to_rgba(ORGAN_COLORS.get(organ_name, "#ffffff"))
        overlay[mask, 0] = r
        overlay[mask, 1] = g
        overlay[mask, 2] = b
        overlay[mask, 3] = alpha
    ax.imshow(overlay, origin="lower", aspect=aspect, interpolation="nearest")


def agreement_map(
    real_seg2d: np.ndarray,
    gen_seg2d: np.ndarray,
    selected_ids: list[tuple[str, int]],
) -> np.ndarray:
    ids = np.array([label_id for _, label_id in selected_ids], dtype=np.int32)
    real_mask = np.isin(real_seg2d, ids)
    gen_mask = np.isin(gen_seg2d, ids)
    out = np.zeros(real_seg2d.shape, dtype=np.uint8)
    out[real_mask & ~gen_mask] = 1
    out[~real_mask & gen_mask] = 2
    out[real_mask & gen_mask] = 3
    return out


def organs_present_in_sample(
    seg3d: np.ndarray,
    selected_ids: list[tuple[str, int]],
) -> list[str]:
    return [name for name, lid in selected_ids if (seg3d == lid).any()]


def plot_one_patient(sample: dict, out_path: Path, organ_names: list[str]) -> None:
    pid = sample["patient_id"]
    selected_ids = sample["selected_ids"]
    zooms = sample["zooms"]
    view_names = ["axial", "coronal", "sagittal"]
    col_titles = ["Real + TotalSeg", "Generated + TotalSeg", "Agreement map"]

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(14, 10),
        gridspec_kw={
            "height_ratios": [4, 2, 2, 0.4],
            "wspace": 0.04,
            "hspace": 0.35,
        },
    )

    all_present: set[str] = set()
    for name in organs_present_in_sample(sample["real_seg"], selected_ids):
        all_present.add(name)

    for view_idx, view in enumerate(view_names):
        ax_real = axes[view_idx, 0]
        ax_gen = axes[view_idx, 1]
        ax_agree = axes[view_idx, 2]

        slice_idx = sample["slice_idx"][view]
        aspect = get_aspect(view, zooms)

        real_ct = display2d(slice2d(sample["real_ct"], view, slice_idx))
        gen_ct = display2d(slice2d(sample["gen_ct"], view, slice_idx))
        real_seg = slice2d(sample["real_seg"], view, slice_idx)
        gen_seg = slice2d(sample["gen_seg"], view, slice_idx)
        agree = display2d(agreement_map(real_seg, gen_seg, selected_ids))

        kw_ct = dict(cmap="gray", vmin=0, vmax=1, origin="lower", aspect=aspect)
        ax_real.imshow(real_ct, **kw_ct)
        ax_gen.imshow(gen_ct, **kw_ct)
        ax_agree.imshow(
            agree,
            cmap=AGREEMENT_CMAP,
            vmin=0,
            vmax=3,
            origin="lower",
            aspect=aspect,
            interpolation="nearest",
        )

        draw_organ_overlays(ax_real, real_seg, selected_ids, aspect=aspect)
        draw_organ_overlays(ax_gen, gen_seg, selected_ids, aspect=aspect)

        for ax in (ax_real, ax_gen, ax_agree):
            ax.set_xticks([])
            ax.set_yticks([])

        ax_real.set_ylabel(f"{view}\nidx={slice_idx}", fontsize=8, labelpad=4)

        for ax, title in zip((ax_real, ax_gen, ax_agree), col_titles):
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

    for col in range(3):
        axes[3, col].set_visible(False)

    organ_patches = [
        mpatches.Patch(
            facecolor=ORGAN_COLORS[org],
            edgecolor="white",
            linewidth=0.5,
            label=org.replace("_", " "),
        )
        for org in organ_names if org in all_present
    ]
    agree_patches = [
        mpatches.Patch(color="#ef476f", label="real only"),
        mpatches.Patch(color="#118ab2", label="generated only"),
        mpatches.Patch(color="#06d6a0", label="overlap"),
    ]

    fig.legend(
        handles=organ_patches,
        title="Organs",
        title_fontsize=9,
        loc="lower left",
        ncol=max(1, min(len(organ_patches), 6)),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.01, -0.01),
    )
    fig.legend(
        handles=agree_patches,
        title="Agreement map",
        title_fontsize=9,
        loc="lower right",
        ncol=3,
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        bbox_to_anchor=(0.99, -0.01),
    )

    fig.suptitle(
        f"Real vs Generated - TotalSegmentator | Patient: {pid}",
        fontsize=13,
        y=1.005,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root", required=True)
    ap.add_argument("--gen_root", required=True)
    ap.add_argument("--real_seg_root", required=True)
    ap.add_argument("--gen_seg_root", required=True)
    ap.add_argument("--test_json", default="dataset/unet_test_data_volumes.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint_tag", default=None)
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--n_samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        required=True,
        help="Path base. Saves <stem>_<patient_id><suffix> for each patient.",
    )
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
        real_path, gen_path, patient_id = resolve_real_gen_paths(
            image_path, args.real_root, gen_root
        )
        real_seg_path = Path(args.real_seg_root) / patient_id / "totalseg.nii.gz"
        gen_seg_path = gen_seg_root / patient_id / "totalseg.nii.gz"

        missing = []
        if not os.path.exists(real_path):
            missing.append("real CT")
        if not os.path.exists(gen_path):
            missing.append("gen CT")
        if not real_seg_path.exists():
            missing.append("real seg")
        if not gen_seg_path.exists():
            missing.append("gen seg")
        if missing:
            print(f"SKIP {patient_id}: missing {', '.join(missing)}")
            continue

        real_ct, real_zooms = load_nifti_canonical(real_path)
        gen_ct, gen_zooms = load_nifti_canonical(gen_path)
        real_seg, label_map, real_seg_zooms = load_multilabel_seg_canonical(str(real_seg_path))
        gen_seg, _, gen_seg_zooms = load_multilabel_seg_canonical(str(gen_seg_path))

        real_ct = clamp_ct_01(real_ct)
        gen_ct = clamp_ct_01(gen_ct)

        if real_ct.shape != real_seg.shape:
            print(f"SKIP {patient_id}: real CT/seg shape mismatch {real_ct.shape} vs {real_seg.shape}")
            continue
        if gen_ct.shape != gen_seg.shape:
            print(f"SKIP {patient_id}: gen CT/seg shape mismatch {gen_ct.shape} vs {gen_seg.shape}")
            continue
        if real_ct.shape != gen_ct.shape:
            print(f"SKIP {patient_id}: real/gen CT shape mismatch {real_ct.shape} vs {gen_ct.shape}")
            continue

        if not np.allclose(real_zooms, gen_zooms, atol=1e-4):
            print(f"WARN {patient_id}: real/gen CT spacing differs {real_zooms} vs {gen_zooms}")
        if not np.allclose(real_zooms, real_seg_zooms, atol=1e-4):
            print(f"WARN {patient_id}: real CT/seg spacing differs {real_zooms} vs {real_seg_zooms}")
        if not np.allclose(gen_zooms, gen_seg_zooms, atol=1e-4):
            print(f"WARN {patient_id}: gen CT/seg spacing differs {gen_zooms} vs {gen_seg_zooms}")

        selected_ids = build_selected_label_ids(label_map, organ_names)
        real_mask = build_union_mask(real_seg, selected_ids)
        slice_idx = best_slice_indices(real_mask)

        samples.append({
            "patient_id": patient_id,
            "real_ct": real_ct,
            "gen_ct": gen_ct,
            "real_seg": real_seg,
            "gen_seg": gen_seg,
            "selected_ids": selected_ids,
            "slice_idx": slice_idx,
            "zooms": real_zooms,
        })

    if not samples:
        raise SystemExit("No valid samples found.")

    out_path = Path(args.output)
    for sample in samples:
        pid = sample["patient_id"]
        p = out_path.parent / f"{out_path.stem}_{pid}{out_path.suffix}"
        plot_one_patient(sample, p, organ_names)

    print(f"\nDone. Saved {len(samples)} figures in {out_path.parent}")


if __name__ == "__main__":
    main()
