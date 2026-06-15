#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import pandas as pd


def atomic_write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def load_ids_from_jsonlist(json_path: Path) -> set[str]:
    j = json.loads(json_path.read_text())

    # supporta vari key usate nei tuoi json
    if "training" in j:
        items = j["training"]
    elif "validation" in j:
        items = j["validation"]
    elif "test" in j:
        items = j["test"]
    else:
        items = []

    return set(Path(x["image"]).parent.name for x in items)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_csv", required=True, help="dataset/reports.csv (VolumeName, Findings_EN, Impressions_EN)")
    ap.add_argument("--train_json", required=True, help="dataset/unet_train_data_volumes.json")
    ap.add_argument("--val_json", required=True, help="dataset/unet_val_data_volumes.json")
    ap.add_argument("--test_json", required=True, help="dataset/unet_test_data_volumes.json")
    ap.add_argument("--pds_root", required=True, help="/home/.../data/private_ct")
    ap.add_argument("--lat_root", required=True, help="dataset/pds_latents")
    ap.add_argument("--out_dir", default="dataset", help="where to write train_reports.csv / val_reports.csv")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing sidecars")
    args = ap.parse_args()

    reports_csv = Path(args.reports_csv)
    train_json = Path(args.train_json)
    val_json = Path(args.val_json)
    test_json   = Path(args.test_json)
    pds_root = Path(args.pds_root).resolve()
    lat_root = Path(args.lat_root).resolve()
    out_dir = Path(args.out_dir)

    df = pd.read_csv(reports_csv)
    if "VolumeName" not in df.columns:
        raise RuntimeError("reports.csv must contain column 'VolumeName'")
    # index rapido per ID
    df["VolumeName"] = df["VolumeName"].astype(str).str.strip()
    rep_map = df.set_index("VolumeName").to_dict(orient="index")

    train_ids = load_ids_from_jsonlist(train_json)
    val_ids = load_ids_from_jsonlist(val_json)
    test_ids = load_ids_from_jsonlist(test_json)
    all_ids = sorted(train_ids | val_ids | test_ids)


    print("TRAIN ids:", len(train_ids))
    print("VAL ids  :", len(val_ids))
    print("TEST ids :", len(test_ids))
    print("TOTAL ids:", len(all_ids))

    # --- split reports in train/val (utile per step embeddings testo)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train = df[df["VolumeName"].isin(train_ids)].copy()
    df_val = df[df["VolumeName"].isin(val_ids)].copy()
    df_test = df[df["VolumeName"].isin(test_ids)].copy()
    train_rep_path = out_dir / "train_reports.csv"
    val_rep_path = out_dir / "val_reports.csv"
    test_rep_path = out_dir / "test_reports.csv"
    df_train.to_csv(train_rep_path, index=False)
    df_val.to_csv(val_rep_path, index=False)
    df_test.to_csv(test_rep_path, index=False)
    print("WROTE:", train_rep_path)
    print("WROTE:", val_rep_path)
    print("WROTE:", test_rep_path)

    # --- sidecars
    missing_latents = 0
    missing_reports = 0
    written = 0
    skipped = 0

    for vid in all_ids:
        # ricostruisco path originale e path latent dal pattern usato nello script:
        # pds_root/<year>/nii/<VID>/ct.nii.gz
        year = vid[3:7] if vid.startswith("IEO") and len(vid) >= 7 else None
        if year is None:
            # fallback: cerca in pds_root
            orig = next(pds_root.glob(f"**/{vid}/ct.nii.gz"), None)
            if orig is None:
                orig = next(pds_root.glob(f"**/{vid}/ct_preprocessed.nii.gz"), None)

            if orig is None:
                print("[WARN] cannot find original for", vid)
                continue
        else:
            orig = pds_root / year / "nii" / vid / "ct.nii.gz"
            if not orig.exists():
                # fallback glob
                orig = next(pds_root.glob(f"**/{vid}/ct.nii.gz"), None)
                if orig is None:
                    orig = next(pds_root.glob(f"**/{vid}/ct_preprocessed.nii.gz"), None)
                if orig is None:
                    print("[WARN] cannot find original for", vid)
                    continue

        latent = Path(str(orig)).as_posix().replace(str(pds_root), str(lat_root))
        latent = Path(latent)

        if not latent.exists():
            missing_latents += 1
            continue

        sidecar = Path(str(latent) + ".json")  # ct.nii.gz.json

        if sidecar.exists() and not args.overwrite:
            skipped += 1
            continue

        # spacing dal nifti originale
        img = nib.load(str(orig))
        zooms = img.header.get_zooms()[:3]
        spacing = [float(zooms[0]), float(zooms[1]), float(zooms[2])]

        # impression dal reports.csv
        rep = rep_map.get(vid)
        if rep is None:
            missing_reports += 1
            impression = ""
        else:
            imp = rep.get("Impressions_EN", "")
            if pd.isna(imp):
                imp = ""
            impression = str(imp)

        payload = {
            "volume_id": vid,
            "spacing": spacing,
            "impression": impression,
        }
        atomic_write_json(sidecar, payload)
        written += 1

    print("\nDONE")
    print("Written sidecars:", written)
    print("Skipped existing :", skipped)
    print("Missing latents  :", missing_latents)
    print("Missing reports  :", missing_reports)
    print("Expected sidecars total (train+val+test):", len(all_ids))



if __name__ == "__main__":
    main()
