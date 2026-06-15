#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from pathlib import Path


def read_metadata_paths(metadata_csv: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    with metadata_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            patient_id = str(row["patient_id"]).strip()
            mask_path = Path(str(row["tumor_mask_path"]).strip())
            if patient_id and patient_id not in paths:
                paths[patient_id] = mask_path
    return paths


def load_split_ids(paths_and_keys: list[tuple[Path, str]]) -> set[str]:
    ids: set[str] = set()
    for path, key in paths_and_keys:
        data = json.loads(path.read_text())
        ids.update(Path(item["image"]).parent.name for item in data.get(key, []))
    return ids


def patient_id_from_source(path: Path) -> str | None:
    match = re.fullmatch(r"(IEO[0-9]+)_ovseg\.nii\.gz", path.name)
    if not match:
        return None
    return match.group(1)


def resolve_repo_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def relative_symlink_target(source: Path, target: Path) -> str:
    return os.path.relpath(source.resolve(), start=target.parent.resolve())


def place_file(source: Path, target: Path, mode: str, overwrite: bool, dry_run: bool) -> str:
    if target.exists() or target.is_symlink():
        if not overwrite:
            return "skip_exists"
        if not dry_run:
            target.unlink()

    if dry_run:
        return f"would_{mode}"

    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(relative_symlink_target(source, target))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return mode


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient_id", "source_path", "target_path", "status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Place flat OVSeg masks into metadata tumor_mask_path locations.")
    parser.add_argument("--source_dir", default="data/private_ovseg_masks")
    parser.add_argument("--metadata_csv", default="data/metadata.csv")
    parser.add_argument("--manifest_csv", default="results/promptgen/prepare_ovseg_masks_manifest.csv")
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--only_split_ids", action="store_true")
    parser.add_argument("--train_json", default="dataset/unet_train_data_volumes.json")
    parser.add_argument("--val_json", default="dataset/unet_val_data_volumes.json")
    parser.add_argument("--test_json", default="dataset/unet_test_data_volumes.json")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    source_dir = resolve_repo_path(Path(args.source_dir), repo_root)
    metadata_csv = resolve_repo_path(Path(args.metadata_csv), repo_root)
    manifest_csv = resolve_repo_path(Path(args.manifest_csv), repo_root)

    if not source_dir.exists():
        raise SystemExit(f"source_dir not found: {source_dir}")
    if not metadata_csv.exists():
        raise SystemExit(f"metadata_csv not found: {metadata_csv}")

    expected_paths = read_metadata_paths(metadata_csv)
    split_ids: set[str] | None = None
    if args.only_split_ids:
        split_ids = load_split_ids(
            [
                (resolve_repo_path(Path(args.train_json), repo_root), "training"),
                (resolve_repo_path(Path(args.val_json), repo_root), "validation"),
                (resolve_repo_path(Path(args.test_json), repo_root), "test"),
            ]
        )

    source_files = sorted(source_dir.glob("*_ovseg.nii.gz"))
    source_by_patient: dict[str, Path] = {}
    ignored_name_rows: list[dict[str, str]] = []
    for source in source_files:
        patient_id = patient_id_from_source(source)
        if patient_id is None:
            ignored_name_rows.append(
                {
                    "patient_id": "",
                    "source_path": str(source),
                    "target_path": "",
                    "status": "ignored",
                    "reason": "filename does not match IEO*_ovseg.nii.gz",
                }
            )
            continue
        source_by_patient[patient_id] = source

    target_ids = set(expected_paths)
    if split_ids is not None:
        target_ids &= split_ids

    rows: list[dict[str, str]] = []
    for patient_id in sorted(target_ids):
        source = source_by_patient.get(patient_id)
        target_rel = expected_paths[patient_id]
        target = resolve_repo_path(target_rel, repo_root)
        if source is None:
            rows.append(
                {
                    "patient_id": patient_id,
                    "source_path": "",
                    "target_path": str(target),
                    "status": "missing_source",
                    "reason": "no source mask for patient_id",
                }
            )
            continue

        status = place_file(source, target, mode=args.mode, overwrite=args.overwrite, dry_run=args.dry_run)
        rows.append(
            {
                "patient_id": patient_id,
                "source_path": str(source),
                "target_path": str(target),
                "status": status,
                "reason": "",
            }
        )

    extra_ids = sorted(set(source_by_patient) - target_ids)
    for patient_id in extra_ids:
        rows.append(
            {
                "patient_id": patient_id,
                "source_path": str(source_by_patient[patient_id]),
                "target_path": "",
                "status": "ignored",
                "reason": "source patient_id not requested by metadata/split filter",
            }
        )
    rows.extend(ignored_name_rows)

    write_manifest(manifest_csv, rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("Source masks:", len(source_by_patient))
    print("Target patients:", len(target_ids))
    print("Status counts:", counts)
    print("Manifest:", manifest_csv)
    if any(row["status"] == "missing_source" for row in rows):
        raise SystemExit("Some target patients are missing source masks; see manifest.")


if __name__ == "__main__":
    main()
