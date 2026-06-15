#!/usr/bin/env python3
"""
Confronto feature radiomiche shape/first-order tra segmentazioni reali e generate.
Output: CSV con feature affiancate, differenze e percentuali.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
import time
import glob

# Plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_ORGANS = [
    "liver", "spleen", "pancreas", "stomach",
    "duodenum", "small_bowel", "colon",
    "kidney_left", "kidney_right", "urinary_bladder",
    "uterus", "rectum",
]

# Funzione per caricare label_map da totalseg

def load_label_map(totalseg_path):
    try:
        from totalsegmentator.nifti_ext_header import load_multilabel_nifti
    except Exception as e:
        raise SystemExit(
            "Cannot import TotalSegmentator nifti_ext_header loader. "
            "Install totalsegmentator in this environment.\n"
            f"Detail: {repr(e)}"
        )
    seg_img, label_map = load_multilabel_nifti(str(totalseg_path))
    return label_map

# Ricava organ_to_id come in evaluate_semantic_consistency

def build_organ_to_id(label_map, organs):
    name_to_id = {str(v): int(k) for k, v in label_map.items()}
    out = {}
    for organ in organs:
        if organ in name_to_id:
            out[organ] = name_to_id[organ]
    return out


def resample_mask_to_image(mask_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
    """Resample mask into the physical space of ref_img using nearest-neighbour."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(mask_img.GetPixelID())
    return resampler.Execute(mask_img)

def extract_features(ct_path, mask_path, organ_id):
    ct_img = sitk.ReadImage(str(ct_path))
    mask_img = sitk.ReadImage(str(mask_path))
    mask_bin = sitk.BinaryThreshold(mask_img, organ_id, organ_id, 1, 0)

    # ── FIX: align mask to CT space if spacing/origin differ ──────────────────
    ct_spacing = ct_img.GetSpacing()
    mask_spacing = mask_bin.GetSpacing()
    if ct_spacing != mask_spacing or ct_img.GetOrigin() != mask_bin.GetOrigin():
        mask_bin = resample_mask_to_image(mask_bin, ct_img)
    # ──────────────────────────────────────────────────────────────────────────

    arr = sitk.GetArrayViewFromImage(mask_bin)
    if arr.sum() == 0:
        return None

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableImageTypeByName("Original")
    extractor.enableFeatureClassByName("shape")
    extractor.enableFeatureClassByName("firstorder")
    features = extractor.execute(ct_img, mask_bin)

    keys = [
        "original_shape_Sphericity", "original_shape_Elongation", "original_shape_Flatness",
        "original_shape_SurfaceArea", "original_shape_Volume",
        "original_firstorder_Mean", "original_firstorder_StdDeviation",
        "original_firstorder_Skewness", "original_firstorder_Kurtosis"
    ]
    return {k: float(features[k]) for k in keys if k in features}

def main():
    ap = argparse.ArgumentParser(description="Radiomics feature comparison: real vs generated")
    ap.add_argument("--real_seg_root", required=True)
    ap.add_argument("--gen_seg_root", required=True)
    ap.add_argument("--checkpoint_tag", required=True)
    ap.add_argument("--ct_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--plot_dir", default=None, help="Directory to save plots (optional)")
    args = ap.parse_args()

    organs = [x.strip() for x in args.organs.split(",") if x.strip()]
    real_patients = sorted([d.name for d in Path(args.real_seg_root).iterdir() if d.is_dir()])
    print(f"[INFO] Patients found: {len(real_patients)}")
    if not real_patients:
        raise SystemExit("No patients found in real_seg_root")
    first_totalseg = Path(args.real_seg_root) / real_patients[0] / "totalseg.nii.gz"
    print(f"[INFO] Loading label_map from: {first_totalseg}")
    if not first_totalseg.exists():
        raise SystemExit(f"totalseg.nii.gz not found: {first_totalseg}")
    label_map = load_label_map(first_totalseg)
    organ_to_id = build_organ_to_id(label_map, organs)
    print(f"[INFO] Organs evaluated: {list(organ_to_id.keys())}")

    # Build a mapping: patient_id -> CT path (search recursively)
    print("[INFO] Building patient-to-CT mapping (recursive search)...")
    ct_map = {}
    ct_patterns = [
        Path(args.ct_root) / "*" / "nii" / "*" / "ct.nii.gz",
        Path(args.ct_root) / "*" / "nii" / "*" / "ct_preprocessed.nii.gz",
    ]
    for pattern in ct_patterns:
        for ct_file in glob.glob(str(pattern), recursive=True):
            pid = Path(ct_file).parent.name
            ct_map.setdefault(pid, ct_file)
    print(f"[INFO] Found CTs for {len(ct_map)} patients.")

    rows = []
    start_time = time.time()
    for idx, pid in enumerate(real_patients):
        ct_path = Path(ct_map.get(pid, ""))
        real_mask = Path(args.real_seg_root) / pid / "totalseg.nii.gz"
        gen_mask = Path(args.gen_seg_root) / args.checkpoint_tag / pid / "totalseg.nii.gz"
        start_patient = time.time()
        print(f"[CHECK] Patient {pid} paths:")
        print(f"        CT:        {ct_path}  Exists: {ct_path.exists()}")
        print(f"        Real seg: {real_mask}  Exists: {real_mask.exists()}")
        print(f"        Gen seg:  {gen_mask}  Exists: {gen_mask.exists()}")
        if not ct_path.exists() or not real_mask.exists() or not gen_mask.exists():
            print(f"[WARN] Skipping {pid}: missing file(s)")
            continue
        print(f"[INFO] Patient {idx+1}/{len(real_patients)}: {pid}")
        for organ in organs:
            organ_id = organ_to_id.get(organ)
            if organ_id is None:
                print(f"[WARN] Organ '{organ}' not found in label_map, skipping.")
                continue
            real_feat = extract_features(ct_path, real_mask, organ_id)
            gen_feat = extract_features(ct_path, gen_mask, organ_id)
            if real_feat is None and gen_feat is None:
                print(f"[WARN] Organ '{organ}' absent in both, skipping.")
                continue
            row = {
                "patient_id": pid,
                "organ": organ,
            }
            for k in real_feat or gen_feat:
                row[f"real_{k}"] = real_feat[k] if real_feat else np.nan
                row[f"gen_{k}"] = gen_feat[k] if gen_feat else np.nan
                row[f"diff_{k}"] = (gen_feat[k] if gen_feat else np.nan) - (real_feat[k] if real_feat else np.nan)
                row[f"perc_diff_{k}"] = (row[f"diff_{k}"] / row[f"real_{k}"] if row[f"real_{k}"] else np.nan)
            rows.append(row)
        print(f"[TIME] Patient {pid} processed in {time.time() - start_patient:.2f} seconds.")

    elapsed = time.time() - start_time
    print(f"[INFO] Extraction completed in {elapsed:.1f} seconds. Writing CSV...")
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] CSV saved: {args.out_csv} (rows: {len(df)})")
    print(f"[CHECK] CSV exists: {Path(args.out_csv).exists()}")

    # Automatic plotting
    if args.plot_dir:
        print(f"[INFO] Generating plots in {args.plot_dir}...")
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        feature_keys = [k for k in df.columns if k.startswith("diff_")]
        for fk in feature_keys:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x="organ", y=fk, data=df)
            plt.title(f"Boxplot difference {fk.replace('diff_', '')}")
            plt.ylabel("Difference (gen - real)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plot_dir / f"boxplot_{fk}.png")
            plt.close()
            print(f"[CHECK] Plot saved: {plot_dir / f'boxplot_{fk}.png'} Exists: {(plot_dir / f'boxplot_{fk}.png').exists()}")
        feature_base = [k.replace("real_", "") for k in df.columns if k.startswith("real_")]
        for fb in feature_base:
            real_col = f"real_{fb}"
            gen_col = f"gen_{fb}"
            if real_col in df.columns and gen_col in df.columns:
                plt.figure(figsize=(7, 7))
                sns.scatterplot(x=df[real_col], y=df[gen_col], hue=df["organ"])
                minval = min(df[real_col].min(), df[gen_col].min())
                maxval = max(df[real_col].max(), df[gen_col].max())
                plt.plot([minval, maxval], [minval, maxval], 'k--', lw=1)
                plt.xlabel("Real feature")
                plt.ylabel("Generated feature")
                plt.title(f"Scatter {fb}: real vs generated")
                plt.tight_layout()
                plt.savefig(plot_dir / f"scatter_{fb}.png")
                plt.close()
                print(f"[CHECK] Plot saved: {plot_dir / f'scatter_{fb}.png'} Exists: {(plot_dir / f'scatter_{fb}.png').exists()}")
        print(f"[INFO] Plots saved in: {plot_dir}")
        print(f"[CHECK] Plot directory exists: {plot_dir.exists()}")

if __name__ == "__main__":
    main()
