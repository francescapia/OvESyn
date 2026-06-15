#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt

try:
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti
except Exception as e:
    raise SystemExit(
        "Errore: non riesco a importare totalsegmentator.nifti_ext_header.load_multilabel_nifti.\n"
        "Esegui nell'env dove TotalSegmentator è installato.\n"
        f"Dettaglio: {repr(e)}"
    )

DEFAULT_ORGANS = [
    "liver", "spleen", "pancreas", "stomach",
    "duodenum", "small_bowel", "colon",
    "kidney_left", "kidney_right", "urinary_bladder",
    "uterus", "rectum",
]

def bbox_from_mask(mask: np.ndarray):
    if not mask.any():
        return None
    xs = np.where(mask.any(axis=(1, 2)))[0]
    ys = np.where(mask.any(axis=(0, 2)))[0]
    zs = np.where(mask.any(axis=(0, 1)))[0]
    return int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1]), int(zs[0]), int(zs[-1])

def crop_slices_from_bbox(bbox, shape, pad_vox):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox
    px, py, pz = pad_vox
    x0 = max(0, xmin - px); x1 = min(shape[0] - 1, xmax + px)
    y0 = max(0, ymin - py); y1 = min(shape[1] - 1, ymax + py)
    z0 = max(0, zmin - pz); z1 = min(shape[2] - 1, zmax + pz)
    return slice(x0, x1 + 1), slice(y0, y1 + 1), slice(z0, z1 + 1)

def mask_volume_ml(mask: np.ndarray, voxel_sizes_mm):
    voxel_vol_mm3 = float(voxel_sizes_mm[0] * voxel_sizes_mm[1] * voxel_sizes_mm[2])
    return float(mask.sum() * voxel_vol_mm3 / 1000.0)

def hu_stats(ct: np.ndarray, mask: np.ndarray):
    vals = ct[mask]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(vals)), float(np.percentile(vals, 10)), float(np.percentile(vals, 90))

def compute_organs_contact(
    tumor_mask: np.ndarray,
    ct_voxel_sizes,
    totalseg_path: Path,
    organs_req,
    topk: int,
    min_voxels: int,
    margin_mm: float,
):
    # load totalseg + label map
    seg_img, label_map = load_multilabel_nifti(str(totalseg_path))
    seg = seg_img.get_fdata(dtype=np.float32).astype(np.int32)
    name_to_id = {v: k for k, v in label_map.items()}

    present_organs = []
    for org in organs_req:
        if org not in name_to_id:
            continue
        oid = int(name_to_id[org])
        vox = int((seg == oid).sum())
        if vox >= min_voxels:
            present_organs.append(org)

    bbox = bbox_from_mask(tumor_mask)
    if bbox is None:
        return ""

    pad_vox = (
        int(np.ceil(margin_mm / ct_voxel_sizes[0])),
        int(np.ceil(margin_mm / ct_voxel_sizes[1])),
        int(np.ceil(margin_mm / ct_voxel_sizes[2])),
    )
    slx, sly, slz = crop_slices_from_bbox(bbox, tumor_mask.shape, pad_vox)

    tum_crop = tumor_mask[slx, sly, slz]
    dt_to_tumor = distance_transform_edt(~tum_crop, sampling=ct_voxel_sizes)

    organ_rows = []
    contact_organs = []

    for org in present_organs:
        oid = int(name_to_id[org])
        org_crop = (seg[slx, sly, slz] == oid)

        if not org_crop.any():
            continue

        contact = bool(np.any(org_crop & tum_crop))
        if contact:
            contact_organs.append(org)
            organ_rows.append({"organ": org, "min_dist_mm": 0.0})
        else:
            min_dist = float(np.min(dt_to_tumor[org_crop]))
            organ_rows.append({"organ": org, "min_dist_mm": min_dist})

    # if any contacts -> return those; else return closest topk
    if contact_organs:
        chosen = contact_organs
    else:
        organ_rows_sorted = sorted(organ_rows, key=lambda r: r["min_dist_mm"])
        chosen = [r["organ"] for r in organ_rows_sorted[:max(1, topk)]]

    return ",".join(chosen)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_voxels", type=int, default=50)
    ap.add_argument("--margin_mm", type=float, default=120.0)
    args = ap.parse_args()

    organs_req = [x.strip() for x in args.organs.split(",") if x.strip()]

    df = pd.read_csv(args.in_csv).fillna("")
    out_rows = []

    required_cols = ["patient_id", "ct_path", "tumor_mask_path", "totalseg_path"]
    for c in required_cols:
        if c not in df.columns:
            raise SystemExit(f"Colonna mancante in {args.in_csv}: '{c}'")

    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        ct_path = Path(row["ct_path"])
        tumor_path = Path(row["tumor_mask_path"])
        totalseg_path = Path(row["totalseg_path"])

        # safety: if files missing -> keep row unchanged
        if not (ct_path.exists() and tumor_path.exists() and totalseg_path.exists()):
            out_rows.append(row.to_dict())
            continue

        # load tumor labels
        tum_data = nib.load(str(tumor_path)).get_fdata(dtype=np.float32).astype(np.int32)
        has1 = bool((tum_data == 1).any())
        has9 = bool((tum_data == 9).any())

        # If NOT both labels -> keep as is
        if not (has1 and has9):
            out_rows.append(row.to_dict())
            continue

        # Otherwise split into two rows (label 1 and label 9), recomputing stats
        ct_img = nib.load(str(ct_path))
        ct = ct_img.get_fdata(dtype=np.float32)
        voxel_sizes = ct_img.header.get_zooms()[:3]

        for lbl in (1, 9):
            mask = (tum_data == lbl)
            if mask.sum() == 0:
                continue

            vol_ml = mask_volume_ml(mask, voxel_sizes)
            mean_hu, p10_hu, p90_hu = hu_stats(ct, mask)

            organs_contact = compute_organs_contact(
                tumor_mask=mask,
                ct_voxel_sizes=voxel_sizes,
                totalseg_path=totalseg_path,
                organs_req=organs_req,
                topk=args.topk,
                min_voxels=args.min_voxels,
                margin_mm=args.margin_mm,
            )

            newrow = row.to_dict()
            newrow["label"] = int(lbl)
            newrow["volume_ml"] = float(vol_ml)
            newrow["mean_hu"] = float(mean_hu)
            newrow["p10_hu"] = float(p10_hu)
            newrow["p90_hu"] = float(p90_hu)
            newrow["organs_contact"] = organs_contact
            out_rows.append(newrow)

    out_df = pd.DataFrame(out_rows)

    # Mantieni ordine colonne come input (stesso schema)
    out_df = out_df[df.columns.tolist()]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print("Rows in:", len(df), "Rows out:", len(out_df))

if __name__ == "__main__":
    main()
