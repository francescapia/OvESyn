#!/usr/bin/env python3
"""Make a paper-ready t-SNE plot for real vs synthetic VAE features."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def load_features(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")
    arr = np.load(path)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2D features in {path}, got shape {arr.shape}")
    return np.nan_to_num(arr.reshape(arr.shape[0], -1).astype(np.float32), copy=False)


def load_json_list(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


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
    if "VolumeName" in df.columns:
        name_col = "VolumeName"
    else:
        name_col = df.columns[0]

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


def resolve_ordered_volume_names(real_metadata: list[dict], gen_metadata: list[dict], n: int) -> list[str]:
    names: list[str] = []
    for i in range(n):
        candidates: Iterable[object] = (
            real_metadata[i].get("path") if i < len(real_metadata) else "",
            gen_metadata[i].get("real_path") if i < len(gen_metadata) else "",
            gen_metadata[i].get("gen_path") if i < len(gen_metadata) else "",
        )
        name = next((volume_name_from_any(c) for c in candidates if str(c or "").strip()), f"case_{i:04d}")
        names.append(name)
    return names


def compute_tsne(x: np.ndarray, seed: int, perplexity: float | None, n_iter: int) -> np.ndarray:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    x = StandardScaler().fit_transform(x)
    n_samples, n_features = x.shape
    n_pca = max(2, min(50, n_features, n_samples - 1))
    x_pca = PCA(n_components=n_pca, random_state=seed).fit_transform(x)

    if perplexity is None:
        perplexity = min(30.0, max(5.0, (n_samples - 1) / 3.0))
    perplexity = min(float(perplexity), max(1.0, n_samples - 2.0))

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
        metric="euclidean",
    )
    try:
        return TSNE(max_iter=n_iter, **kwargs).fit_transform(x_pca)
    except TypeError:
        return TSNE(n_iter=n_iter, **kwargs).fit_transform(x_pca)


def plot_panel(ax, df: pd.DataFrame, label_col: str, title: str, palette: dict[str, str], connect_pairs: bool) -> None:
    if connect_pairs:
        real = df[df["domain"] == "Real"].set_index("pair_id")
        synthetic = df[df["domain"] == "Synthetic"].set_index("pair_id")
        for pair_id in sorted(set(real.index) & set(synthetic.index)):
            ax.plot(
                [real.loc[pair_id, "tsne_1"], synthetic.loc[pair_id, "tsne_1"]],
                [real.loc[pair_id, "tsne_2"], synthetic.loc[pair_id, "tsne_2"]],
                color="#a8a8a8",
                linewidth=0.35,
                alpha=0.25,
                zorder=1,
            )

    for label, color in palette.items():
        for domain, marker, size, alpha, edge in (
            ("Real", "o", 42, 0.90, "#202020"),
            ("Synthetic", "^", 50, 0.74, "#ffffff"),
        ):
            sub = df[(df[label_col] == label) & (df["domain"] == domain)]
            if sub.empty:
                continue
            ax.scatter(
                sub["tsne_1"],
                sub["tsne_2"],
                s=size,
                marker=marker,
                c=color,
                edgecolors=edge,
                linewidths=0.45,
                alpha=alpha,
                zorder=3 if domain == "Real" else 2,
            )

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#fbfbfa")


def add_legends(fig, axes, figo_palette: dict[str, str], ascites_palette: dict[str, str]) -> None:
    from matplotlib.lines import Line2D

    domain_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#bfbfbf", markeredgecolor="#202020",
               markersize=6, label="Real"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#bfbfbf", markeredgecolor="#ffffff",
               markersize=7, label="Synthetic"),
    ]
    figo_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none",
               markersize=6, label=label)
        for label, color in figo_palette.items()
    ]
    ascites_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="none",
               markersize=6, label=label.replace("Ascites ", ""))
        for label, color in ascites_palette.items()
    ]

    axes[0].legend(handles=figo_handles + domain_handles, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                   ncol=4, frameon=False, fontsize=7, handletextpad=0.35, columnspacing=0.9)
    axes[1].legend(handles=ascites_handles + domain_handles, loc="lower center", bbox_to_anchor=(0.5, -0.18),
                   ncol=4, frameon=False, fontsize=7, handletextpad=0.35, columnspacing=0.9)


def make_plot(df: pd.DataFrame, out_prefix: Path, connect_pairs: bool, dpi: int) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })

    figo_palette = {
        "FIGO III": "#007c89",
        "FIGO IV": "#d1495b",
        "Unknown": "#9a9a9a",
    }
    ascites_palette = {
        "Ascites absent": "#2f5aa6",
        "Ascites present": "#d97904",
        "Unknown": "#9a9a9a",
    }
    figo_palette = {k: v for k, v in figo_palette.items() if (df["figo"] == k).any()}
    ascites_palette = {k: v for k, v in ascites_palette.items() if (df["ascites"] == k).any()}

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.2), constrained_layout=True)
    plot_panel(axes[0], df, "figo", "FIGO stage", figo_palette, connect_pairs)
    plot_panel(axes[1], df, "ascites", "Ascites", ascites_palette, connect_pairs)
    add_legends(fig, axes, figo_palette, ascites_palette)

    fig.savefig(out_prefix.with_suffix(".pdf"))
    fig.savefig(out_prefix.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_path: Path, args: argparse.Namespace) -> None:
    summary = {
        "n_points": int(len(df)),
        "n_pairs": int(df["pair_id"].nunique()),
        "real_features": str(args.real_features),
        "synthetic_features": str(args.gen_features),
        "reports_csv": str(args.reports_csv),
        "seed": args.seed,
        "perplexity": args.perplexity,
        "n_iter": args.n_iter,
        "counts": {
            "domain": df["domain"].value_counts().to_dict(),
            "figo": df[df["domain"] == "Real"]["figo"].value_counts().to_dict(),
            "ascites": df[df["domain"] == "Real"]["ascites"].value_counts().to_dict(),
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot real-vs-synthetic t-SNE from cached VAE features.")
    ap.add_argument("--real_features", type=Path, required=True)
    ap.add_argument("--gen_features", type=Path, required=True)
    ap.add_argument("--real_metadata", type=Path, required=True)
    ap.add_argument("--gen_metadata", type=Path, required=True)
    ap.add_argument("--reports_csv", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--out_name", default="tsne_real_vs_synthetic_labels")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--perplexity", type=float, default=None)
    ap.add_argument("--n_iter", type=int, default=1500)
    ap.add_argument("--dpi", type=int, default=450)
    ap.add_argument("--connect_pairs", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    real_features = load_features(args.real_features)
    gen_features = load_features(args.gen_features)
    real_metadata = load_json_list(args.real_metadata)
    gen_metadata = load_json_list(args.gen_metadata)
    labels = load_report_labels(args.reports_csv)

    n = min(len(real_features), len(gen_features), len(real_metadata), len(gen_metadata))
    if n <= 2:
        raise ValueError(f"Need at least 3 paired samples, got {n}")
    if len({len(real_features), len(gen_features), len(real_metadata), len(gen_metadata)}) != 1:
        print(
            "Warning: input lengths differ; truncating to "
            f"{n} pairs (real_features={len(real_features)}, gen_features={len(gen_features)}, "
            f"real_metadata={len(real_metadata)}, gen_metadata={len(gen_metadata)})."
        )

    volume_names = resolve_ordered_volume_names(real_metadata, gen_metadata, n)
    x = np.concatenate([real_features[:n], gen_features[:n]], axis=0)
    coords = compute_tsne(x, seed=args.seed, perplexity=args.perplexity, n_iter=args.n_iter)

    rows = []
    for i, name in enumerate(volume_names):
        row_labels = labels.get(name, {"figo": "Unknown", "ascites": "Unknown"})
        rows.append({
            "pair_id": i,
            "volume_name": name,
            "domain": "Real",
            "figo": row_labels["figo"],
            "ascites": row_labels["ascites"],
            "tsne_1": coords[i, 0],
            "tsne_2": coords[i, 1],
        })
        rows.append({
            "pair_id": i,
            "volume_name": name,
            "domain": "Synthetic",
            "figo": row_labels["figo"],
            "ascites": row_labels["ascites"],
            "tsne_1": coords[n + i, 0],
            "tsne_2": coords[n + i, 1],
        })

    df = pd.DataFrame(rows)
    out_prefix = args.out_dir / args.out_name
    df.to_csv(out_prefix.with_name(out_prefix.name + "_coordinates.csv"), index=False)
    write_summary(df, out_prefix.with_name(out_prefix.name + "_summary.json"), args)
    make_plot(df, out_prefix, connect_pairs=args.connect_pairs, dpi=args.dpi)
    print(f"Saved {out_prefix.with_suffix('.pdf')}")
    print(f"Saved {out_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
