#!/usr/bin/env python3
"""Plot per-volume synthetic fidelity stratified by conditioning labels."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_SPECS = {
    "int_wasserstein": {
        "title": "Intensity W1",
        "ylabel": "Wasserstein-1",
    },
    "int_delta_mean": {
        "title": "Mean intensity",
        "ylabel": "Absolute mean error",
    },
    "int_delta_std": {
        "title": "Intensity spread",
        "ylabel": "Absolute std. error",
    },
}

LABEL_SPECS = {
    "figo": {
        "row_title": "FIGO stage",
        "order": ["FIGO III", "FIGO IV"],
        "palette": {
            "FIGO III": "#007c89",
            "FIGO IV": "#d1495b",
            "Unknown": "#9a9a9a",
        },
    },
    "ascites": {
        "row_title": "Ascites",
        "order": ["Ascites absent", "Ascites present"],
        "palette": {
            "Ascites absent": "#2f5aa6",
            "Ascites present": "#d97904",
            "Unknown": "#9a9a9a",
        },
    },
}


def volume_name_from_any(value: object) -> str:
    text = str(value or "")
    match = re.search(r"\bIEO\d+\b", text)
    if match:
        return match.group(0)

    path = Path(text)
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    else:
        name = path.stem
    if name in {"ct", "ct_preprocessed"} and path.parent.name:
        return path.parent.name
    return name


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_figo(text: str) -> str:
    low = text.lower()
    if re.search(r"\bfigo\s*(stage\s*)?(iv|4)\b|\bstage\s*(iv|4)\s*disease\b", low):
        return "FIGO IV"
    if re.search(r"\bfigo\s*(stage\s*)?(iii|3)\b|\bstage\s*(iii|3)\s*disease\b", low):
        return "FIGO III"
    return "Unknown"


def extract_ascites(text: str) -> str:
    low = text.lower()
    absent = (
        r"\bno\s+ascites\b",
        r"\bascites\s+is\s+absent\b",
        r"\babsence\s+of\s+ascites\b",
        r"\bwithout\s+ascites\b",
        r"\babsent\s+ascites\b",
    )
    present = (
        r"\bascites\s+is\s+present\b",
        r"\bascites\s+present\b",
        r"\bascites\s+noted\b",
        r"\bwith\s+ascites\b",
        r"\bpresence\s+of\s+ascites\b",
    )
    if any(re.search(pattern, low) for pattern in absent):
        return "Ascites absent"
    if any(re.search(pattern, low) for pattern in present):
        return "Ascites present"
    return "Unknown"


def load_report_labels(reports_csv: Path) -> dict[str, dict[str, str]]:
    if not reports_csv.exists():
        raise FileNotFoundError(f"Missing reports CSV: {reports_csv}")

    df = pd.read_csv(reports_csv).fillna("")
    name_col = "VolumeName" if "VolumeName" in df.columns else df.columns[0]
    text_cols = [c for c in ("Findings_EN", "Impressions_EN", "Findings", "Impressions") if c in df.columns]
    if not text_cols:
        text_cols = [c for c in df.columns if c != name_col]

    labels: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        name = volume_name_from_any(row[name_col])
        text = " ".join(normalize_text(row[c]) for c in text_cols)
        labels[name] = {
            "figo": extract_figo(text),
            "ascites": extract_ascites(text),
        }
    return labels


def load_one_metrics_csv(metrics_csv: Path, model_name: str | None, source_index: int) -> pd.DataFrame:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_csv}")

    df = pd.read_csv(metrics_csv)
    if model_name and "model" in df.columns:
        df = df[df["model"].astype(str) == model_name].copy()
    if df.empty:
        raise ValueError(f"No rows found in {metrics_csv} after model filter: {model_name}")

    if "model" not in df.columns:
        df["model"] = model_name or "model"
    if "idx" not in df.columns:
        df["idx"] = np.arange(len(df))

    missing_metrics = [m for m in METRIC_SPECS if m not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing required metric columns in {metrics_csv}: {missing_metrics}")

    source_col = "real_path" if "real_path" in df.columns else "gen_path"
    if source_col not in df.columns:
        raise ValueError(f"Could not find real_path or gen_path in {metrics_csv}")

    df["source_csv"] = str(metrics_csv)
    df["source_index"] = source_index
    return df


def average_metrics_if_needed(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if len(frames) == 1:
        out = frames[0].copy()
        out["n_generations"] = 1
        return out

    df = pd.concat(frames, ignore_index=True)
    group_cols = ["idx", "real_path", "model"]
    agg = {
        "gen_path": "first",
        "source_csv": lambda x: ";".join(sorted(set(map(str, x)))),
        "source_index": "nunique",
        **{m: "mean" for m in METRIC_SPECS},
    }
    out = df.groupby(group_cols, as_index=False).agg(agg)
    out = out.rename(columns={"source_index": "n_generations"})
    return out


def load_metrics(metrics_csvs: list[Path], reports_csv: Path, model_name: str | None) -> pd.DataFrame:
    frames = [
        load_one_metrics_csv(metrics_csv, model_name=model_name, source_index=i)
        for i, metrics_csv in enumerate(metrics_csvs)
    ]
    df = average_metrics_if_needed(frames)

    source_col = "real_path" if "real_path" in df.columns else "gen_path"
    labels = load_report_labels(reports_csv)
    df["volume_name"] = df[source_col].map(volume_name_from_any)
    df["figo"] = df["volume_name"].map(lambda x: labels.get(x, {}).get("figo", "Unknown"))
    df["ascites"] = df["volume_name"].map(lambda x: labels.get(x, {}).get("ascites", "Unknown"))
    return df


def ordered_labels(df: pd.DataFrame, label_col: str, include_unknown: bool) -> list[str]:
    spec = LABEL_SPECS[label_col]
    order = [label for label in spec["order"] if (df[label_col] == label).any()]
    extras = sorted(
        label for label in df[label_col].dropna().unique()
        if label not in order and (include_unknown or label != "Unknown")
    )
    return order + extras


def draw_distribution_panel(
    ax,
    df: pd.DataFrame,
    label_col: str,
    metric: str,
    labels: list[str],
    y_max: float,
    rng: np.random.Generator,
) -> None:
    spec = LABEL_SPECS[label_col]
    palette = spec["palette"]
    values = [
        df.loc[df[label_col] == label, metric].dropna().astype(float).to_numpy()
        for label in labels
    ]
    positions = np.arange(len(labels), dtype=float)

    nonempty_values = [v for v in values if len(v)]
    if nonempty_values:
        violins = ax.violinplot(
            nonempty_values,
            positions=[positions[i] for i, v in enumerate(values) if len(v)],
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        body_idx = 0
        for i, vals in enumerate(values):
            if not len(vals):
                continue
            color = palette.get(labels[i], "#6f6f6f")
            body = violins["bodies"][body_idx]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.20)
            body.set_linewidth(0.9)
            body_idx += 1

    for i, vals in enumerate(values):
        color = palette.get(labels[i], "#6f6f6f")
        if not len(vals):
            continue
        jitter = rng.normal(0.0, 0.055, size=len(vals))
        ax.scatter(
            positions[i] + jitter,
            vals,
            s=16,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            zorder=3,
        )

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.plot([positions[i] - 0.18, positions[i] + 0.18], [med, med], color="#202020", lw=1.4, zorder=4)
        ax.plot([positions[i], positions[i]], [q1, q3], color="#202020", lw=2.8, alpha=0.85, zorder=4)
        ax.text(
            positions[i],
            y_max * 0.965,
            f"n={len(vals)}",
            ha="center",
            va="top",
            fontsize=6.8,
            color="#4a4a4a",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([label.replace("Ascites ", "") for label in labels], fontsize=7)
    ax.set_xlim(-0.55, max(0.55, len(labels) - 0.45))
    ax.set_ylim(0.0, y_max)
    ax.grid(axis="y", color="#dfdfdf", linewidth=0.55, alpha=0.65)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7, length=0)
    ax.tick_params(axis="x", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#fbfbfa")


def make_summary(df: pd.DataFrame, include_unknown: bool) -> pd.DataFrame:
    rows = []
    for label_col in LABEL_SPECS:
        labels = ordered_labels(df, label_col, include_unknown)
        for label in labels:
            sub = df[df[label_col] == label]
            for metric in METRIC_SPECS:
                vals = sub[metric].dropna().astype(float).to_numpy()
                rows.append({
                    "condition": label_col,
                    "label": label,
                    "metric": metric,
                    "n": len(vals),
                    "mean": float(np.mean(vals)) if len(vals) else np.nan,
                    "median": float(np.median(vals)) if len(vals) else np.nan,
                    "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)) if len(vals) else np.nan,
                    "q25": float(np.percentile(vals, 25)) if len(vals) else np.nan,
                    "q75": float(np.percentile(vals, 75)) if len(vals) else np.nan,
                })
    return pd.DataFrame(rows)


def make_plot(df: pd.DataFrame, out_prefix: Path, include_unknown: bool, seed: int, dpi: int) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    metrics = list(METRIC_SPECS)
    conditions = list(LABEL_SPECS)
    y_max_by_metric: dict[str, float] = {}
    for metric in metrics:
        vals = df[metric].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            y_max_by_metric[metric] = 1.0
            continue
        ymax = float(np.nanmax(vals))
        y_max_by_metric[metric] = max(ymax * 1.14, 1e-6)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(
        len(conditions),
        len(metrics),
        figsize=(7.3, 4.55),
        constrained_layout=True,
        sharey="col",
    )

    for col_idx, metric in enumerate(metrics):
        axes[0, col_idx].set_title(METRIC_SPECS[metric]["title"], fontsize=9, fontweight="bold", pad=8)
        for row_idx, condition in enumerate(conditions):
            labels = ordered_labels(df, condition, include_unknown)
            draw_distribution_panel(
                axes[row_idx, col_idx],
                df,
                condition,
                metric,
                labels,
                y_max_by_metric[metric],
                rng,
            )
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(
                    f"{LABEL_SPECS[condition]['row_title']}\n{METRIC_SPECS[metric]['ylabel']}",
                    fontsize=8,
                    labelpad=8,
                )
            else:
                axes[row_idx, col_idx].set_ylabel(METRIC_SPECS[metric]["ylabel"], fontsize=8)

    fig.suptitle("Class-conditioned real-synthetic fidelity", fontsize=11, fontweight="bold", y=1.03)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot per-volume fidelity metrics grouped by FIGO and ascites.")
    ap.add_argument("--metrics_csv", type=Path, nargs="+", required=True)
    ap.add_argument("--reports_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--out_name", default="class_conditioned_fidelity")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--include_unknown", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dpi", type=int, default=450)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_metrics(args.metrics_csv, args.reports_csv, args.model_name)
    if not args.include_unknown:
        df = df[(df["figo"] != "Unknown") & (df["ascites"] != "Unknown")].copy()
    if df.empty:
        raise ValueError("No rows remain after dropping unknown labels.")

    out_prefix = args.out_dir / args.out_name
    summary = make_summary(df, args.include_unknown)
    summary.to_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), index=False)
    df.to_csv(out_prefix.with_name(out_prefix.name + "_labeled_metrics.csv"), index=False)
    if len(args.metrics_csv) > 1:
        df.to_csv(out_prefix.with_name(out_prefix.name + "_averaged_per_volume_metrics.csv"), index=False)

    metadata = {
        "metrics_csv": [str(p) for p in args.metrics_csv],
        "n_metrics_csv": len(args.metrics_csv),
        "reports_csv": str(args.reports_csv),
        "model_name": args.model_name,
        "n_rows": int(len(df)),
        "figo_counts": df["figo"].value_counts().to_dict(),
        "ascites_counts": df["ascites"].value_counts().to_dict(),
    }
    out_prefix.with_name(out_prefix.name + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    make_plot(df, out_prefix, args.include_unknown, args.seed, args.dpi)

    print(f"Saved {out_prefix.with_suffix('.pdf')}")
    print(f"Saved {out_prefix.with_suffix('.png')}")
    print(f"Saved {out_prefix.with_name(out_prefix.name + '_summary.csv')}")


if __name__ == "__main__":
    main()
