#!/usr/bin/env python3
"""
Collect Promptgen v6 experiment metrics into CSV and LaTeX tables.

Example:
  python scripts/collect_paper_metrics.py \
    --run promptgenV6_clipBASE_unetBASE=CLIP base + UNet base \
    --run 2026-06-08_120000_promptgenV6_clipLORA_clipgbs24_run01=CLIP LoRA \
    --out_dir results/paper_tables/promptgen_v6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" in spec:
        tag, label = spec.split("=", 1)
        return tag.strip(), label.strip()
    tag = spec.strip()
    return tag, tag


def read_csvs(paths: Iterable[Path], experiment: str, run_tag: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        df = pd.read_csv(path)
        df.insert(0, "source_csv", path.as_posix())
        df.insert(0, "run_tag", run_tag)
        df.insert(0, "experiment", experiment)
        frames.append(df)
    return frames


def write_table(df: pd.DataFrame, out_csv: Path, out_tex: Path, caption: str, label: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    if df.empty:
        out_tex.write_text("% No rows available yet.\n")
        return
    tex = df.to_latex(
        index=False,
        escape=True,
        float_format=lambda x: f"{x:.4f}",
        caption=caption,
        label=label,
    )
    out_tex.write_text(tex)


def compact_clip_table(df: pd.DataFrame) -> pd.DataFrame:
    wanted = [
        "experiment",
        "model_name",
        "split",
        "loss",
        "retrieval_i2t_R@1",
        "retrieval_i2t_R@5",
        "retrieval_t2i_R@1",
        "retrieval_t2i_R@5",
        "clip_score_diag_mean",
        "probe_figo_3v4_acc",
        "probe_figo_3v4_auc",
        "probe_ascites_acc",
        "probe_ascites_auc",
        "probe_heterogeneity_acc",
        "probe_heterogeneity_auc",
    ]
    cols = [c for c in wanted if c in df.columns]
    return df[cols].copy() if cols else df.copy()


def collect_generation_summary(run_dir: Path, experiment: str, run_tag: str) -> pd.DataFrame:
    image_frames = read_csvs(
        run_dir.glob("evaluation/image_metrics/*/summary_metrics.csv"),
        experiment,
        run_tag,
    )
    semantic_frames = read_csvs(
        run_dir.glob("evaluation/semantic/*/summary_semantic_metrics.csv"),
        experiment,
        run_tag,
    )

    image = pd.concat(image_frames, ignore_index=True) if image_frames else pd.DataFrame()
    semantic = pd.concat(semantic_frames, ignore_index=True) if semantic_frames else pd.DataFrame()

    if not image.empty:
        image["checkpoint_tag"] = image["source_csv"].map(lambda p: Path(p).parent.name)
    if not semantic.empty:
        semantic["checkpoint_tag"] = semantic["source_csv"].map(lambda p: Path(p).parent.name)

    if image.empty:
        return semantic
    if semantic.empty:
        return image

    merge_cols = ["experiment", "run_tag", "checkpoint_tag"]
    return image.merge(
        semantic.drop(columns=["source_csv"], errors="ignore"),
        on=merge_cols,
        how="outer",
        suffixes=("", "_semantic"),
    )


def collect_radiomics(run_dir: Path, experiment: str, run_tag: str) -> pd.DataFrame:
    rows: list[dict] = []
    for csv_path in sorted(run_dir.glob("evaluation/radiomics/compare_radiomics_*.csv")):
        df = pd.read_csv(csv_path)
        checkpoint_tag = csv_path.stem.replace("compare_radiomics_", "").removesuffix("_test")
        for col in [c for c in df.columns if c.startswith("diff_")]:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if values.empty:
                continue
            feature = col.removeprefix("diff_")
            pct_col = f"perc_diff_{feature}"
            pct_values = pd.to_numeric(df[pct_col], errors="coerce").dropna() if pct_col in df else pd.Series(dtype=float)
            rows.append(
                {
                    "experiment": experiment,
                    "run_tag": run_tag,
                    "checkpoint_tag": checkpoint_tag,
                    "feature": feature,
                    "n": int(values.shape[0]),
                    "mean_diff": float(values.mean()),
                    "mean_abs_diff": float(values.abs().mean()),
                    "median_abs_diff": float(values.abs().median()),
                    "mean_abs_percent_diff": float(pct_values.abs().mean()) if not pct_values.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect CLIP and generation metrics into paper-ready tables.")
    ap.add_argument("--pipeline_root", default="runs/clipFT_FT")
    ap.add_argument("--out_dir", default="results/paper_tables/promptgen_v6")
    ap.add_argument("--run", action="append", required=True, help="RUN_TAG=Experiment label")
    args = ap.parse_args()

    pipeline_root = Path(args.pipeline_root)
    out_dir = Path(args.out_dir)

    clip_frames: list[pd.DataFrame] = []
    generation_frames: list[pd.DataFrame] = []
    radiomics_frames: list[pd.DataFrame] = []

    for spec in args.run:
        run_tag, label = parse_run_spec(spec)
        run_dir = pipeline_root / run_tag
        if not run_dir.exists():
            print(f"[warn] missing run directory: {run_dir}")
            continue

        clip_frames.extend(
            read_csvs(run_dir.glob("clip/evaluation/**/clip_eval_test.csv"), label, run_tag)
        )
        generation = collect_generation_summary(run_dir, label, run_tag)
        if not generation.empty:
            generation_frames.append(generation)
        radiomics = collect_radiomics(run_dir, label, run_tag)
        if not radiomics.empty:
            radiomics_frames.append(radiomics)

    clip = pd.concat(clip_frames, ignore_index=True) if clip_frames else pd.DataFrame()
    generation = pd.concat(generation_frames, ignore_index=True) if generation_frames else pd.DataFrame()
    radiomics = pd.concat(radiomics_frames, ignore_index=True) if radiomics_frames else pd.DataFrame()

    write_table(
        clip,
        out_dir / "clip_metrics_full.csv",
        out_dir / "clip_metrics_full.tex",
        "CLIP evaluation metrics on Promptgen v6 reports.",
        "tab:clip_promptgen_v6_full",
    )
    write_table(
        compact_clip_table(clip),
        out_dir / "clip_metrics_compact.csv",
        out_dir / "clip_metrics_compact.tex",
        "Compact CLIP evaluation metrics on Promptgen v6 reports.",
        "tab:clip_promptgen_v6_compact",
    )
    write_table(
        generation,
        out_dir / "generation_metrics.csv",
        out_dir / "generation_metrics.tex",
        "Image and semantic generation metrics on Promptgen v6 reports.",
        "tab:generation_promptgen_v6",
    )
    write_table(
        radiomics,
        out_dir / "radiomics_metrics.csv",
        out_dir / "radiomics_metrics.tex",
        "Radiomics feature differences between real and generated volumes.",
        "tab:radiomics_promptgen_v6",
    )

    print(f"[done] wrote tables under {out_dir}")


if __name__ == "__main__":
    main()
