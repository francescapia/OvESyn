#!/usr/bin/env python3
"""
Visual comparison between real and generated CT volumes using TotalSegmentator overlays.
One PNG per patient. Layout: rows = real/gen/agreement, cols = axial/coronal/sagittal.

Coronal and sagittal are shown in portrait (tall) orientation using aspect=1,
which respects natural pixel proportions: (512,128) → tall portrait.
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
from matplotlib.gridspec import GridSpec
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
    "#ef476f",  # real only
    "#118ab2",  # generated only
    "#06d6a0",  # overlap
])

# Rows = real CT, generated CT, agreement
ROW_LABELS = ["Real + TotalSeg", "Generated + TotalSeg", "Agreement map"]

# Cols = axial, coronal, sagittal
# Width ratios: axial is square (512×512), coronal/sagittal are narrow (128 wide)
# so axial gets 4x more width than coronal/sagittal
COL_WIDTH_RATIOS = [4, 1, 1]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

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
    p1  = float(np.percentile(vol, 1))
    p99 = float(np.percentile(vol, 99))
    if p99 <= p1:
        return np.zeros_like(vol)
    return np.clip((vol - p1) / (p99 - p1), 0.0, 1.0)


def load_multilabel_seg(seg_path: str) -> tuple[np.ndarray, dict]:
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti
    seg_img, label_map = load_multilabel_nifti(seg_path)
    seg = seg_img.get_fdata(dtype=np.float32).astype(np.int32)
    return seg, label_map


def build_selected_label_ids(label_map: dict, organ_names: list[str]) -> list[tuple[str, int]]:
    name_to_id = {v: int(k) for k, v in label_map.items()}
    return [(organ, name_to_id[organ]) for organ in organ_names if organ in name_to_id]


def build_union_mask(seg: np.ndarray, selected_ids: list[tuple[str, int]]) -> np.ndarray:
    if not selected_ids:
        return np.zeros_like(seg, dtype=bool)
    ids = np.array([label_id for _, label_id in selected_ids], dtype=np.int32)
    return np.isin(seg, ids)


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def best_slice_indices(reference_mask: np.ndarray) -> dict[str, int]:
    if not reference_mask.any():
        h, w, d = reference_mask.shape
        return {"axial": d // 2, "coronal": w // 2, "sagittal": h // 2}
    return {
        "axial":    int(np.argmax(reference_mask.sum(axis=(0, 1)))),
        "coronal":  int(np.argmax(reference_mask.sum(axis=(0, 2)))),
        "sagittal": int(np.argmax(reference_mask.sum(axis=(1, 2)))),
    }


def slice2d(vol: np.ndarray, view: str, index: int) -> np.ndarray:
    """
    axial:    (X,Y)  → mostrato con .T     → quadrata
    coronal:  (X,Z)  → NO .T, Z flip      → portrait (craniale in alto)
    sagittal: (Y,Z)  → NO .T, Z flip      → portrait (craniale in alto)
    """
    if view == "axial":
        return vol[:, :, index]
    if view == "coronal":
        return vol[:, index, ::-1]
    return vol[index, :, ::-1]


def show_slice(ax, sl: np.ndarray, view: str, **imshow_kw):
    """
    aspect=1 → pixel quadrati → (512,128) diventa naturalmente portrait alto.
    axial usa .T per avere orientamento standard (L→R, A→P).
    """
    imshow_kw.setdefault("origin", "lower")
    imshow_kw["aspect"] = 1   # sempre pixel quadrati
    if view == "axial":
        ax.imshow(sl.T, **imshow_kw)
    else:
        ax.imshow(sl, **imshow_kw)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_organ_overlays(
    ax,
    seg2d: np.ndarray,
    selected_ids: list[tuple[str, int]],
    view: str,
    alpha: float = 0.45,
) -> None:
    h, w = seg2d.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    for organ_name, label_id in selected_ids:
        mask = (seg2d == label_id)
        if not mask.any():
            continue
        r, g, b, _ = to_rgba(ORGAN_COLORS.get(organ_name, "#ffffff"))
        overlay[mask, 0] = r
        overlay[mask, 1] = g
        overlay[mask, 2] = b
        overlay[mask, 3] = alpha
    if view == "axial":
        ax.imshow(overlay.transpose(1, 0, 2), origin="lower", aspect=1)
    else:
        ax.imshow(overlay, origin="lower", aspect=1)


def agreement_map(
    real_seg2d: np.ndarray,
    gen_seg2d: np.ndarray,
    selected_ids: list[tuple[str, int]],
) -> np.ndarray:
    ids = np.array([label_id for _, label_id in selected_ids], dtype=np.int32)
    real_mask = np.isin(real_seg2d, ids)
    gen_mask  = np.isin(gen_seg2d,  ids)
    out = np.zeros(real_seg2d.shape, dtype=np.uint8)
    out[real_mask & ~gen_mask] = 1
    out[~real_mask & gen_mask] = 2
    out[real_mask & gen_mask]  = 3
    return out


def organs_present_in_sample(
    seg3d: np.ndarray,
    selected_ids: list[tuple[str, int]],
) -> list[str]:
    return [name for name, lid in selected_ids if (seg3d == lid).any()]


# ---------------------------------------------------------------------------
# Per-patient plot
# ---------------------------------------------------------------------------

def plot_one_patient(sample: dict, out_path: Path, organ_names: list[str]) -> None:
    """
    Layout 3×3:
      rows = Real CT | Generated CT | Agreement
      cols = Axial   | Coronal      | Sagittal

    Con aspect=1 e COL_WIDTH_RATIOS=[4,1,1]:
      - Axial occupa il 67% della larghezza → quadrata e grande
      - Coronal e Sagittal occupano il 16% ciascuna → portrait alti
    """
    pid          = sample["patient_id"]
    selected_ids = sample["selected_ids"]
    views        = ["axial", "coronal", "sagittal"]

    # figsize: larghezza 14, altezza proporzionale
    # La riga è alta quanto la vista axial (512px) → le coronal/sagittal
    # con aspect=1 e width=1/6 * 14 ≈ 2.3in → height ≈ 512/128 * 2.3 ≈ 9.2in
    # Ma siamo vincolati dalla griglia → usiamo altezza fissa generosa
    fig = plt.figure(figsize=(14, 13))
    gs  = GridSpec(
        nrows=4,    # 3 righe dati + 1 legenda
        ncols=3,
        figure=fig,
        width_ratios=COL_WIDTH_RATIOS,
        height_ratios=[1, 1, 1, 0.08],
        wspace=0.06,
        hspace=0.28,
    )

    all_present: set[str] = set()
    for name in organs_present_in_sample(sample["real_seg"], selected_ids):
        all_present.add(name)

    # Dati per ogni cella: (row_label, data_key_or_agree)
    row_data = [
        ("real_ct",  "real_seg"),
        ("gen_ct",   "gen_seg"),
        None,  # agreement
    ]

    for view_idx, view in enumerate(views):
        slice_idx = sample["slice_idx"][view]
        real_ct   = slice2d(sample["real_ct"],  view, slice_idx)
        gen_ct    = slice2d(sample["gen_ct"],   view, slice_idx)
        real_seg  = slice2d(sample["real_seg"], view, slice_idx)
        gen_seg   = slice2d(sample["gen_seg"],  view, slice_idx)
        agree     = agreement_map(real_seg, gen_seg, selected_ids)

        slices = [real_ct, gen_ct, agree]
        segs   = [real_seg, gen_seg, None]

        for row_idx, (sl, seg) in enumerate(zip(slices, segs)):
            ax = fig.add_subplot(gs[row_idx, view_idx])

            if row_idx < 2:
                show_slice(ax, sl, view, cmap="gray", vmin=0, vmax=1)
                if seg is not None:
                    draw_organ_overlays(ax, seg, selected_ids, view)
            else:
                show_slice(ax, agree, view, cmap=AGREEMENT_CMAP, vmin=0, vmax=3)

            ax.set_xticks([])
            ax.set_yticks([])

            # Titoli colonne (viste) solo sulla prima riga
            if row_idx == 0:
                ax.set_title(
                    f"{view.capitalize()}\nidx={slice_idx}",
                    fontsize=9, fontweight="bold", pad=4,
                )

            # Label righe solo sulla colonna axial
            if view_idx == 0:
                ax.set_ylabel(ROW_LABELS[row_idx], fontsize=8, labelpad=4)

    # ── Riga legenda ──────────────────────────────────────────────────────────
    for col in range(3):
        fig.add_subplot(gs[3, col]).set_visible(False)

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
        title="Organs", title_fontsize=9,
        loc="lower left", ncol=min(len(organ_patches), 6),
        fontsize=8, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.01, -0.01),
    )
    fig.legend(
        handles=agree_patches,
        title="Agreement map", title_fontsize=9,
        loc="lower right", ncol=3,
        fontsize=8, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.99, -0.01),
    )

    fig.suptitle(
        f"Real vs Generated — TotalSegmentator | Patient: {pid}",
        fontsize=13, y=1.005,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_root",       required=True)
    ap.add_argument("--gen_root",        required=True)
    ap.add_argument("--real_seg_root",   required=True)
    ap.add_argument("--gen_seg_root",    required=True)
    ap.add_argument("--test_json",       default="dataset/unet_test_data_volumes.json")
    ap.add_argument("--split",           default="test")
    ap.add_argument("--checkpoint_tag",  default=None)
    ap.add_argument("--organs",          default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--n_samples",       type=int, default=3)
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--output",          required=True,
                    help="Path base. Salva <stem>_<patient_id><suffix> per ogni paziente.")
    args = ap.parse_args()

    organ_names  = [x.strip() for x in args.organs.split(",") if x.strip()]
    gen_root     = resolve_generated_root(args.gen_root, args.checkpoint_tag)
    gen_seg_root = Path(args.gen_seg_root)
    if args.checkpoint_tag:
        gen_seg_root = gen_seg_root / args.checkpoint_tag

    paths          = load_split_paths(args.test_json, args.split)
    selected_paths = stratified_sample(paths, args.n_samples, args.seed)

    samples = []
    for image_path in selected_paths:
        real_path, gen_path, patient_id = resolve_real_gen_paths(
            image_path, args.real_root, gen_root
        )
        real_seg_path = Path(args.real_seg_root) / patient_id / "totalseg.nii.gz"
        gen_seg_path  = gen_seg_root / patient_id / "totalseg.nii.gz"

        missing = []
        if not os.path.exists(real_path):  missing.append("real CT")
        if not os.path.exists(gen_path):   missing.append("gen CT")
        if not real_seg_path.exists():     missing.append("real seg")
        if not gen_seg_path.exists():      missing.append("gen seg")
        if missing:
            print(f"SKIP {patient_id}: missing {', '.join(missing)}")
            continue

        real_ct             = normalize_ct(load_ct(real_path))
        gen_ct              = normalize_ct(load_ct(gen_path))
        real_seg, label_map = load_multilabel_seg(str(real_seg_path))
        gen_seg, _          = load_multilabel_seg(str(gen_seg_path))

        selected_ids = build_selected_label_ids(label_map, organ_names)
        real_mask    = build_union_mask(real_seg, selected_ids)
        slice_idx    = best_slice_indices(real_mask)

        samples.append({
            "patient_id":   patient_id,
            "real_ct":      real_ct,
            "gen_ct":       gen_ct,
            "real_seg":     real_seg,
            "gen_seg":      gen_seg,
            "selected_ids": selected_ids,
            "slice_idx":    slice_idx,
        })

    if not samples:
        raise SystemExit("No valid samples found.")

    out_path = Path(args.output)
    for sample in samples:
        pid = sample["patient_id"]
        p   = out_path.parent / f"{out_path.stem}_{pid}{out_path.suffix}"
        plot_one_patient(sample, p, organ_names)

    print(f"\nDone. Saved {len(samples)} figures in {out_path.parent}")


if __name__ == "__main__":
    main()