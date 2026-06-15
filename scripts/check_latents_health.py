#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


DEFAULT_JSONS = (
    ("dataset/unet_train_data_volumes.json", "training"),
    ("dataset/unet_val_data_volumes.json", "validation"),
    ("dataset/unet_test_data_volumes.json", "test"),
)


def normalize_path(path: str | os.PathLike) -> str:
    value = Path(path).as_posix()
    if value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def expected_latent_path(image_path: str, data_base_dir: str, latent_root: Path) -> Path:
    image_norm = normalize_path(image_path)
    data_base_norm = normalize_path(data_base_dir)
    preproc_prefix = "data/private_ct_preprocessed"

    rel_path = None
    for prefix in (preproc_prefix, data_base_norm, "data/private_ct"):
        if image_norm.startswith(prefix + "/"):
            rel_path = image_norm[len(prefix) + 1 :]
            break

    if rel_path is None:
        rel_path = image_norm

    return latent_root / rel_path


def load_expected_paths(latent_root: Path, data_base_dir: str) -> list[Path]:
    paths: list[Path] = []
    for json_path, split in DEFAULT_JSONS:
        path = Path(json_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for item in payload.get(split, []):
            paths.append(expected_latent_path(item["image"], data_base_dir, latent_root))
    return sorted(set(paths))


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be comma-separated integers") from exc
    if not shape:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    return shape


def signal_name(returncode: int) -> str:
    if returncode >= 0:
        return str(returncode)
    sig = -returncode
    try:
        return signal.Signals(sig).name
    except ValueError:
        return f"signal_{sig}"


def check_one(args: argparse.Namespace) -> None:
    result = {
        "path": args.check_one,
        "exists": False,
        "sidecar": False,
        "shape": "",
        "dtype": "",
        "finite": False,
        "min": "",
        "max": "",
        "mean": "",
        "std": "",
        "status": "bad",
        "problems": [],
    }

    path = Path(args.check_one)
    result["exists"] = path.exists()
    result["sidecar"] = Path(str(path) + ".json").exists()

    if not path.exists():
        result["problems"].append("missing_latent")
        print(json.dumps(result), flush=True)
        return

    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(path), mmap=False)
        result["shape"] = "x".join(str(dim) for dim in img.shape)
        result["dtype"] = str(img.get_data_dtype())

        arr = img.get_fdata(dtype=np.float32, caching="unchanged")
        result["finite"] = bool(np.isfinite(arr).all())
        result["min"] = f"{float(np.nanmin(arr)):.8g}"
        result["max"] = f"{float(np.nanmax(arr)):.8g}"
        result["mean"] = f"{float(np.nanmean(arr)):.8g}"
        result["std"] = f"{float(np.nanstd(arr)):.8g}"

        expected_shape = tuple(args.expected_shape)
        if expected_shape and tuple(img.shape) != expected_shape:
            result["problems"].append(f"shape={tuple(img.shape)}")
        if not result["finite"]:
            result["problems"].append("nan_or_inf")

        std = float(result["std"])
        mean = float(result["mean"])
        if std == 0:
            result["problems"].append("zero_std")
        if abs(mean) > args.max_abs_mean or std > args.max_std:
            result["problems"].append(f"suspicious_stats_mean={mean:.6g}_std={std:.6g}")
        if args.check_sidecars and not result["sidecar"]:
            result["problems"].append("missing_sidecar")

        result["status"] = "ok" if not result["problems"] else "bad"
    except Exception as exc:
        result["problems"].append(f"unreadable:{type(exc).__name__}:{exc}")

    print(json.dumps(result), flush=True)


def run_child(path: Path, args: argparse.Namespace) -> dict[str, object]:
    cmd = [
        sys.executable,
        __file__,
        "--check-one",
        str(path),
        "--expected-shape",
        ",".join(str(dim) for dim in args.expected_shape),
        "--max-abs-mean",
        str(args.max_abs_mean),
        "--max-std",
        str(args.max_std),
    ]
    if args.check_sidecars:
        cmd.append("--check-sidecars")

    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode == 0 and proc.stdout.strip():
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            pass

    return {
        "path": str(path),
        "exists": path.exists(),
        "sidecar": Path(str(path) + ".json").exists(),
        "shape": "",
        "dtype": "",
        "finite": False,
        "min": "",
        "max": "",
        "mean": "",
        "std": "",
        "status": "bad",
        "problems": [f"child_crashed:{signal_name(proc.returncode)}"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check VAE latent NIfTI files for basic corruption indicators.")
    parser.add_argument("--latent-root", default="dataset/pds_latents_v5_newpreprocessing_clipFT")
    parser.add_argument("--data-base-dir", default="./data/private_ct")
    parser.add_argument("--expected-shape", type=parse_shape, default=(128, 128, 32, 4))
    parser.add_argument("--max-abs-mean", type=float, default=100.0)
    parser.add_argument("--max-std", type=float, default=100.0)
    parser.add_argument("--check-sidecars", action="store_true")
    parser.add_argument("--existing-only", action="store_true", help="Scan existing latent files instead of JSON-expected paths.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--report-csv", default="")
    parser.add_argument("--check-one", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.check_one:
        check_one(args)
        return

    latent_root = Path(args.latent_root)
    if args.existing_only:
        paths = sorted(latent_root.rglob("ct.nii.gz"))
    else:
        paths = load_expected_paths(latent_root, args.data_base_dir)

    if args.limit > 0:
        paths = paths[: args.limit]

    rows = []
    counts = {"ok": 0, "bad": 0, "missing": 0}
    for idx, path in enumerate(paths, 1):
        row = run_child(path, args)
        rows.append(row)
        status = str(row["status"])
        counts["ok" if status == "ok" else "bad"] += 1
        if not row.get("exists"):
            counts["missing"] += 1

        if status != "ok":
            print(f"[{idx}/{len(paths)}] BAD {path}: {';'.join(row['problems'])}", flush=True)
        elif idx % 25 == 0 or idx == len(paths):
            print(f"[{idx}/{len(paths)}] ok={counts['ok']} bad={counts['bad']} missing={counts['missing']}", flush=True)

    print("\nDONE")
    print(f"checked: {len(rows)}")
    print(f"ok: {counts['ok']}")
    print(f"bad: {counts['bad']}")
    print(f"missing: {counts['missing']}")

    report_csv = args.report_csv or str(latent_root / "latent_health_report.csv")
    fieldnames = ["status", "path", "exists", "sidecar", "shape", "dtype", "finite", "min", "max", "mean", "std", "problems"]
    Path(report_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(report_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["problems"] = ";".join(row["problems"])
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"report_csv: {report_csv}")


if __name__ == "__main__":
    main()
