#!/usr/bin/env python3
"""
Semantic consistency evaluation (attribute-level) using TotalSegmentator outputs.

Compares real vs generated segmentations per patient and computes:
- presence agreement per organ
- Dice per organ
- relative volume error per organ
- prompt-mentioned organ consistency (if reports CSV is provided)

Expected segmentation layout:
  real:      <real_seg_root>/<patient_id>/totalseg.nii.gz
  generated: <gen_seg_root>/<checkpoint_tag>/<patient_id>/totalseg.nii.gz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_ORGANS = [
    "liver", "spleen", "pancreas", "stomach",
    "duodenum", "small_bowel", "colon",
    "kidney_left", "kidney_right", "urinary_bladder",
    "uterus", "rectum",
]

ORGAN_ALIASES = {
    "liver": [r"\bliver\b", r"\bhepatic\b"],
    "spleen": [r"\bspleen\b", r"\bsplenic\b"],
    "pancreas": [r"\bpancreas\b", r"\bpancreatic\b"],
    "stomach": [r"\bstomach\b", r"\bgastric\b"],
    "duodenum": [r"\bduodenum\b", r"\bduodenal\b"],
    "small_bowel": [r"\bsmall bowel\b", r"\bsmall_bowel\b"],
    "colon": [r"\bcolon\b", r"\bcolonic\b", r"\blarge bowel\b"],
    "kidney_left": [r"\bleft kidney\b", r"\bkidney_left\b"],
    "kidney_right": [r"\bright kidney\b", r"\bkidney_right\b"],
    "urinary_bladder": [r"\burinary bladder\b", r"\bbladder\b"],
    "uterus": [r"\buterus\b", r"\buterine\b"],
    "rectum": [r"\brectum\b", r"\brectal\b"],
}


def load_split_paths(data_json: str, split: str) -> List[str]:
    with open(data_json) as f:
        data = json.load(f)
    key_map = {
        "train": "training", "training": "training",
        "val": "validation", "validation": "validation",
        "test": "test", "testing": "test",
    }
    key = key_map.get(split, split)
    if key not in data:
        raise KeyError(f"Split '{split}' not found in {data_json}. Available: {list(data.keys())}")
    return [item["image"] for item in data[key]]


def patient_id_from_json_path(p: str) -> str:
    return Path(p).parent.name


def load_multilabel_seg(seg_path: Path):
    try:
        from totalsegmentator.nifti_ext_header import load_multilabel_nifti
    except Exception as e:
        raise SystemExit(
            "Cannot import TotalSegmentator nifti_ext_header loader. "
            "Install totalsegmentator in this environment.\n"
            f"Detail: {repr(e)}"
        )

    seg_img, label_map = load_multilabel_nifti(str(seg_path))
    seg = seg_img.get_fdata(dtype=np.float32).astype(np.int32)
    return seg, label_map


def build_organ_to_id(label_map: Dict, organs: List[str]) -> Dict[str, int]:
    name_to_id = {str(v): int(k) for k, v in label_map.items()}
    out: Dict[str, int] = {}
    for organ in organs:
        if organ in name_to_id:
            out[organ] = name_to_id[organ]
    return out


def extract_mentioned_organs(text: str, organs: List[str]) -> List[str]:
    t = (text or "").lower()
    found = []
    for organ in organs:
        patterns = ORGAN_ALIASES.get(organ, [rf"\b{re.escape(organ)}\b"])
        if any(re.search(pat, t) for pat in patterns):
            found.append(organ)
    return sorted(set(found))


def dice_from_counts(a: int, b: int, inter: int) -> float:
    denom = a + b
    if denom == 0:
        return np.nan
    return float(2.0 * inter / denom)


def evaluate_case(
    patient_id: str,
    real_seg: np.ndarray,
    gen_seg: np.ndarray,
    organ_to_id: Dict[str, int],
    min_voxels: int,
    mentioned_organs: List[str],
) -> Tuple[dict, List[dict]]:
    per_organ_rows: List[dict] = []

    presence_agree = []
    dice_values = []
    rel_vol_errors = []

    mentioned_real_present = set()
    mentioned_gen_present = set()

    for organ, oid in organ_to_id.items():
        real_count = int((real_seg == oid).sum())
        gen_count = int((gen_seg == oid).sum())
        inter_count = int(np.logical_and(real_seg == oid, gen_seg == oid).sum())

        real_present = real_count >= min_voxels
        gen_present = gen_count >= min_voxels

        presence_agree.append(int(real_present == gen_present))

        if real_present or gen_present:
            d = dice_from_counts(real_count, gen_count, inter_count)
            if not np.isnan(d):
                dice_values.append(d)

        if real_count > 0:
            rel_vol_errors.append(abs(gen_count - real_count) / real_count)

        if organ in mentioned_organs:
            if real_present:
                mentioned_real_present.add(organ)
            if gen_present:
                mentioned_gen_present.add(organ)

        per_organ_rows.append({
            "patient_id": patient_id,
            "organ": organ,
            "real_voxels": real_count,
            "gen_voxels": gen_count,
            "intersection_voxels": inter_count,
            "real_present": int(real_present),
            "gen_present": int(gen_present),
            "presence_match": int(real_present == gen_present),
            "dice": dice_from_counts(real_count, gen_count, inter_count),
            "rel_volume_error": float(abs(gen_count - real_count) / max(real_count, 1)),
            "mentioned_in_prompt": int(organ in mentioned_organs),
        })

    # Prompt-mentioned consistency (set overlap between real-present and gen-present within mentioned organs)
    if mentioned_real_present or mentioned_gen_present:
        tp = len(mentioned_real_present & mentioned_gen_present)
        fp = len(mentioned_gen_present - mentioned_real_present)
        fn = len(mentioned_real_present - mentioned_gen_present)
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else np.nan
    else:
        prec = rec = f1 = np.nan

    case_row = {
        "patient_id": patient_id,
        "n_organs_evaluated": len(organ_to_id),
        "presence_agreement": float(np.mean(presence_agree)) if presence_agree else np.nan,
        "dice_mean": float(np.mean(dice_values)) if dice_values else np.nan,
        "rel_volume_error_mean": float(np.mean(rel_vol_errors)) if rel_vol_errors else np.nan,
        "mentioned_organs_count": int(len(mentioned_organs)),
        "mentioned_presence_precision": float(prec) if not np.isnan(prec) else np.nan,
        "mentioned_presence_recall": float(rec) if not np.isnan(rec) else np.nan,
        "mentioned_presence_f1": float(f1) if not np.isnan(f1) else np.nan,
    }

    return case_row, per_organ_rows


def main():
    ap = argparse.ArgumentParser(description="Semantic consistency evaluation from TotalSegmentator outputs")
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--real_seg_root", required=True)
    ap.add_argument("--gen_seg_root", required=True)
    ap.add_argument("--checkpoint_tag", required=True)
    ap.add_argument("--reports_csv", default=None,
                    help="Optional CSV with prompt/report text; expects VolumeName and Findings_EN")
    ap.add_argument("--organs", default=",".join(DEFAULT_ORGANS))
    ap.add_argument("--min_voxels", type=int, default=50)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    organs = [x.strip() for x in args.organs.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_map: Dict[str, str] = {}
    if args.reports_csv:
        df_reports = pd.read_csv(args.reports_csv).fillna("")
        if "VolumeName" in df_reports.columns and "Findings_EN" in df_reports.columns:
            report_map = dict(zip(df_reports["VolumeName"].astype(str), df_reports["Findings_EN"].astype(str)))

    split_paths = load_split_paths(args.test_json, args.split)
    patient_ids = [patient_id_from_json_path(p) for p in split_paths]

    per_case_rows: List[dict] = []
    per_organ_rows: List[dict] = []
    skipped_rows: List[dict] = []

    organ_to_id_ref: Dict[str, int] | None = None

    for pid in patient_ids:
        real_seg_path = Path(args.real_seg_root) / pid / "totalseg.nii.gz"
        gen_seg_path = Path(args.gen_seg_root) / args.checkpoint_tag / pid / "totalseg.nii.gz"

        if not real_seg_path.exists() or not gen_seg_path.exists():
            skipped_rows.append({
                "patient_id": pid,
                "real_exists": int(real_seg_path.exists()),
                "gen_exists": int(gen_seg_path.exists()),
                "reason": "missing segmentation",
            })
            continue

        try:
            real_seg, real_label_map = load_multilabel_seg(real_seg_path)
            gen_seg, _ = load_multilabel_seg(gen_seg_path)
        except Exception as e:
            skipped_rows.append({
                "patient_id": pid,
                "real_exists": int(real_seg_path.exists()),
                "gen_exists": int(gen_seg_path.exists()),
                "reason": f"load_error: {repr(e)}",
            })
            continue

        if real_seg.shape != gen_seg.shape:
            skipped_rows.append({
                "patient_id": pid,
                "real_exists": 1,
                "gen_exists": 1,
                "reason": f"shape_mismatch real={real_seg.shape} gen={gen_seg.shape}",
            })
            continue

        if organ_to_id_ref is None:
            organ_to_id_ref = build_organ_to_id(real_label_map, organs)

        mentioned = extract_mentioned_organs(report_map.get(pid, ""), organs)
        case_row, organ_rows = evaluate_case(
            patient_id=pid,
            real_seg=real_seg,
            gen_seg=gen_seg,
            organ_to_id=organ_to_id_ref,
            min_voxels=args.min_voxels,
            mentioned_organs=mentioned,
        )
        per_case_rows.append(case_row)
        per_organ_rows.extend(organ_rows)

    df_case = pd.DataFrame(per_case_rows)
    df_organ = pd.DataFrame(per_organ_rows)
    df_skipped = pd.DataFrame(skipped_rows)

    if not df_case.empty:
        summary = {
            "n_cases_evaluated": int(len(df_case)),
            "n_cases_skipped": int(len(df_skipped)),
            "presence_agreement_mean": float(df_case["presence_agreement"].mean()),
            "dice_mean": float(df_case["dice_mean"].mean()),
            "rel_volume_error_mean": float(df_case["rel_volume_error_mean"].mean()),
            "mentioned_presence_f1_mean": float(df_case["mentioned_presence_f1"].mean(skipna=True)),
        }
    else:
        summary = {
            "n_cases_evaluated": 0,
            "n_cases_skipped": int(len(df_skipped)),
            "presence_agreement_mean": np.nan,
            "dice_mean": np.nan,
            "rel_volume_error_mean": np.nan,
            "mentioned_presence_f1_mean": np.nan,
        }

    pd.DataFrame([summary]).to_csv(out_dir / "summary_semantic_metrics.csv", index=False)
    df_case.to_csv(out_dir / "per_volume_semantic_metrics.csv", index=False)
    df_organ.to_csv(out_dir / "per_organ_semantic_metrics.csv", index=False)
    df_skipped.to_csv(out_dir / "skipped_cases.csv", index=False)

    if not df_organ.empty:
        df_organ_summary = (
            df_organ.groupby("organ", as_index=False)
            .agg(
                dice_mean=("dice", "mean"),
                rel_volume_error_mean=("rel_volume_error", "mean"),
                presence_match_rate=("presence_match", "mean"),
                n_cases=("patient_id", "count"),
            )
            .sort_values("organ")
        )
        df_organ_summary.to_csv(out_dir / "organ_summary.csv", index=False)

    print(f"Saved semantic evaluation outputs in: {out_dir}")
    print(f"Cases evaluated: {summary['n_cases_evaluated']} | skipped: {summary['n_cases_skipped']}")
    print(
        "Means -> "
        f"presence_agreement={summary['presence_agreement_mean']:.4f}, "
        f"dice={summary['dice_mean']:.4f}, "
        f"rel_volume_error={summary['rel_volume_error_mean']:.4f}, "
        f"mentioned_f1={summary['mentioned_presence_f1_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
