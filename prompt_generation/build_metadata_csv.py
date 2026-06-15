#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import distance_transform_edt
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


try:
    # serve per leggere totalseg + label_map (nome organo -> id)
    from totalsegmentator.nifti_ext_header import load_multilabel_nifti
except Exception as e:
    raise SystemExit(
        "Errore: non riesco a importare totalsegmentator.nifti_ext_header.load_multilabel_nifti.\n"
        "Esegui questo script nell'env dove TotalSegmentator è installato (es. conda activate totalseg).\n"
        f"Dettaglio: {repr(e)}"
    )

DEFAULT_ORGANS = [
    "liver", "spleen", "pancreas", "stomach",
    "duodenum", "small_bowel", "colon",
    "kidney_left", "kidney_right", "urinary_bladder",
    "uterus", "rectum",  # opzionali, se non nel label_map vengono ignorati
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

def resolve_totalseg_path(case: str, totalseg_root: Path) -> Path | None:
    cand = totalseg_root / case / "totalseg.nii.gz"
    if cand.exists():
        return cand
    return None

def label_from_tumor_mask(tum_data: np.ndarray):
    # decide "label" per compatibilità con codice dottorando
    vals = set(np.unique(tum_data.astype(np.int32)).tolist())
    if 9 in vals:
        return 9
    if 1 in vals:
        return 1
    return 9

def mask_volume_ml(mask: np.ndarray, voxel_sizes_mm):
    voxel_vol_mm3 = float(voxel_sizes_mm[0] * voxel_sizes_mm[1] * voxel_sizes_mm[2])
    return float(mask.sum() * voxel_vol_mm3 / 1000.0)

def hu_stats(ct: np.ndarray, mask: np.ndarray):
    vals = ct[mask]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    p10 = float(np.percentile(vals, 10))
    p90 = float(np.percentile(vals, 90))
    return mean, p10, p90

def skip(case: str, reason: str, **extra):
    d = {"_skip": True, "patient_id": case, "reason": reason}
    d.update(extra)
    return d

def process_one_case(ct_path_str: str, totalseg_root_str: str, ovseg_rel: str,
                     tumor_labels: list[int], organs_req: list[str],
                     topk: int, min_voxels: int, margin_mm: float):
    ct_path = Path(ct_path_str)
    case = ct_path.parent.name

    tumor_path = ct_path.parent / ovseg_rel
    if not tumor_path.exists():
        return {"_skip": True, "patient_id": case, "reason": "missing tumor mask", "tumor_mask_path": str(tumor_path)}

    totalseg_root = Path(totalseg_root_str)
    totalseg_path = resolve_totalseg_path(case, totalseg_root)
    if totalseg_path is None:
        return {"_skip": True, "patient_id": case, "reason": "missing totalseg path"}

    try:


        # --- load CT for HU + voxel sizes ---
        ct_img = nib.load(str(ct_path))
        ct = ct_img.get_fdata(dtype=np.float32)
        voxel_sizes = ct_img.header.get_zooms()[:3]  # mm

        # --- load tumor mask ---
        tum_img = nib.load(str(tumor_path))
        tum_data = tum_img.get_fdata(dtype=np.float32).astype(np.int32)
        if ct.shape != tum_data.shape:
            return {"_skip": True, "patient_id": case, "reason": "shape mismatch CT vs tumor mask",
                    "ct_shape": str(ct.shape), "tumor_shape": str(tum_data.shape),
                    "ct_path": str(ct_path), "tumor_mask_path": str(tumor_path)}

        tum_mask = np.isin(tum_data, tumor_labels)

        if tum_mask.sum() == 0:
            return {"_skip": True, "patient_id": case, "reason": "tumor mask empty (0 voxels for labels 1/9)", "tumor_mask_path": str(tumor_path)}

        # label heuristic (for location in generator code)
        lbl = label_from_tumor_mask(tum_data)

        # volume + HU stats
        vol_ml = mask_volume_ml(tum_mask, voxel_sizes)
        mean_hu, p10_hu, p90_hu = hu_stats(ct, tum_mask)

        # --- load TotalSeg with label_map ---
        seg_img, label_map = load_multilabel_nifti(str(totalseg_path))
        seg = seg_img.get_fdata(dtype=np.float32).astype(np.int32)
        if seg.shape != ct.shape:
            return {"_skip": True, "patient_id": case, "reason": "shape mismatch CT vs TotalSeg",
                    "ct_shape": str(ct.shape), "totalseg_shape": str(seg.shape),
                    "ct_path": str(ct_path), "totalseg_path": str(totalseg_path)}

        name_to_id = {v: k for k, v in label_map.items()}

        # present organs
        present_organs = []
        for org in organs_req:
            if org not in name_to_id:
                continue
            oid = int(name_to_id[org])
            vox = int((seg == oid).sum())
            if vox >= min_voxels:
                present_organs.append(org)

        # crop around tumor for speed (come vostro fast script)
        tum_bbox = bbox_from_mask(tum_mask)
        if tum_bbox is None:
            return {"_skip": True, "patient_id": case, "reason": "bbox_from_mask returned None (unexpected)", "tumor_mask_path": str(tumor_path)}


        pad_vox = (
            int(np.ceil(margin_mm / voxel_sizes[0])),
            int(np.ceil(margin_mm / voxel_sizes[1])),
            int(np.ceil(margin_mm / voxel_sizes[2])),
        )
        slx, sly, slz = crop_slices_from_bbox(tum_bbox, tum_mask.shape, pad_vox)
        tum_crop = tum_mask[slx, sly, slz]

        # distance transform to tumor (0 inside tumor, >0 outside)
        dt_to_tumor = distance_transform_edt(~tum_crop, sampling=voxel_sizes)

        organ_rows = []
        contact_organs = []

        for org in present_organs:
            oid = int(name_to_id[org])
            org_mask = (seg == oid)
            org_crop = org_mask[slx, sly, slz]

            contact = bool(np.any(org_crop & tum_crop))
            if contact:
                min_dist = 0.0
                contact_organs.append(org)
            else:
                if org_crop.any():
                    min_dist = float(np.min(dt_to_tumor[org_crop]))
                else:
                    # se l'organo non cade nel crop, non lo usiamo
                    continue

            organ_rows.append({"organ": org, "contact": contact, "min_dist_mm": min_dist})

        organ_rows_sorted = sorted(organ_rows, key=lambda r: r["min_dist_mm"])

        # organs_contact: prima contatto, altrimenti closest topk
        if contact_organs:
            organs_contact = contact_organs
        else:
            organs_contact = [r["organ"] for r in organ_rows_sorted[:max(1, topk)]]

        return {
            "patient_id": case,
            "label": int(lbl),
            "volume_ml": float(vol_ml),
            "mean_hu": float(mean_hu),
            "p10_hu": float(p10_hu),
            "p90_hu": float(p90_hu),
            "organs_contact": ",".join(organs_contact),
            "ct_path": str(ct_path),
            "tumor_mask_path": str(tumor_path),
            "totalseg_path": str(totalseg_path),
        }

    except Exception as e:
        return {
            "_skip": True,
            "patient_id": case,
            "reason": f"exception: {type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "ct_path": str(ct_path),
            "tumor_mask_path": str(tumor_path),
            "totalseg_path": str(totalseg_path) if totalseg_path else None,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct_json", required=True, help="JSON stile unet_train_data_volumes.json (key 'training' con 'image').")
    ap.add_argument("--totalseg_root", required=True, help="Root con totalseg_output/<CASE>/totalseg.nii.gz")
    ap.add_argument("--ovseg_rel", default="ovseg_predictions_pod_om/tumor_mask_labels_1_9.nii.gz")
    ap.add_argument("--tumor_labels", default="1,9")
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_voxels", type=int, default=50)
    ap.add_argument("--margin_mm", type=float, default=120.0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--max_cases", type=int, default=0)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    tumor_labels = [int(x.strip()) for x in args.tumor_labels.split(",") if x.strip()]
    organs_req = [x.strip() for x in args.organs.split(",") if x.strip()]

    with open(args.ct_json, "r") as f:
        j = json.load(f)
    split_key = "training" if "training" in j else ("validation" if "validation" in j else "test")
    ct_paths = [Path(it["image"]) for it in j.get(split_key, [])]
    ct_paths = [p for p in ct_paths if p.exists()]

    if args.max_cases and args.max_cases > 0:
        ct_paths = ct_paths[:args.max_cases]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []

    with ProcessPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futs = [
            ex.submit(
                process_one_case,
                str(p),
                args.totalseg_root,
                args.ovseg_rel,
                tumor_labels,
                organs_req,
                args.topk,
                args.min_voxels,
                args.margin_mm,
            )
            for p in ct_paths
        ]
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                skipped.append({
                    "_skip": True,
                    "patient_id": "UNKNOWN",
                    "reason": f"worker failed: {type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                continue

            if r is None:
                skipped.append({"_skip": True, "patient_id": "UNKNOWN", "reason": "returned None"})
                continue

            if isinstance(r, dict) and r.get("_skip"):
                skipped.append(r)
            else:
                rows.append(r)


    df = pd.DataFrame(rows).sort_values("patient_id")
    df.to_csv(out_csv, index=False)
    skipped_csv = out_csv.with_name(out_csv.stem + "_skipped.csv")
    pd.DataFrame(skipped).to_csv(skipped_csv, index=False)
    print("Saved skipped log:", skipped_csv)
    print("Cases skipped:", len(skipped))

    print("Saved:", out_csv)
    print("Cases written:", len(df))

if __name__ == "__main__":
    main()
