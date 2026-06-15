from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


LEGACY_DATA_PREFIX = "data/private_ct"
PREPROCESSED_DATA_PREFIX = "data/private_ct_preprocessed"


@dataclass(frozen=True)
class EmbeddingItem:
    split: str
    image_path: str
    volume_name: str
    embedding_path: Path


def load_volume_list(json_path: Path) -> list[str]:
    with open(json_path, "r") as file:
        data = json.load(file)
    key = "training" if "training" in data else ("validation" if "validation" in data else "test")
    return [_item["image"] for _item in data.get(key, [])]


def strip_nii_gz(path: str) -> str:
    return path[:-7] if path.endswith(".nii.gz") else Path(path).with_suffix("").as_posix()


def build_embedding_path(image_path: str, data_base_dir: Path, embedding_base_dir: Path, encoder_name: str) -> Path:
    img_path = Path(image_path)
    rel_str = img_path.as_posix().lstrip("./")

    base_prefixes = [
        data_base_dir.as_posix().lstrip("./").rstrip("/"),
        LEGACY_DATA_PREFIX,
        PREPROCESSED_DATA_PREFIX,
    ]
    for base_prefix in dict.fromkeys(base_prefixes):
        if base_prefix and rel_str.startswith(base_prefix + "/"):
            rel_str = rel_str[len(base_prefix) + 1 :]
            break

    base = strip_nii_gz(rel_str)
    return embedding_base_dir / f"{base}_impression_{encoder_name}.npy"


def load_items(
    split: str,
    json_path: Path,
    data_base_dir: Path,
    embedding_base_dir: Path,
    encoder_name: str,
) -> list[EmbeddingItem]:
    items: list[EmbeddingItem] = []
    for image_path in load_volume_list(json_path):
        emb_path = build_embedding_path(image_path, data_base_dir, embedding_base_dir, encoder_name)
        items.append(
            EmbeddingItem(
                split=split,
                image_path=image_path,
                volume_name=Path(image_path).parent.name,
                embedding_path=emb_path,
            )
        )
    return items


def deranged_indices(n_items: int, rng: random.Random, allow_identity: bool) -> list[int]:
    indices = list(range(n_items))
    if allow_identity:
        rng.shuffle(indices)
        return indices

    if n_items < 2:
        raise ValueError("Cannot create a no-identity shuffle with fewer than two items.")

    for _ in range(1000):
        rng.shuffle(indices)
        if all(source_idx != target_idx for target_idx, source_idx in enumerate(indices)):
            return indices

    shift = rng.randrange(1, n_items)
    return [(idx + shift) % n_items for idx in range(n_items)]


def check_unique_targets(items: list[EmbeddingItem], split: str) -> None:
    seen: set[Path] = set()
    duplicates: list[Path] = []
    for item in items:
        if item.embedding_path in seen:
            duplicates.append(item.embedding_path)
        seen.add(item.embedding_path)
    if duplicates:
        preview = "\n".join(path.as_posix() for path in duplicates[:20])
        raise RuntimeError(f"Duplicate target embedding paths in split={split}:\n{preview}")


def shuffle_split(
    split: str,
    json_path: Path,
    source_dir: Path,
    target_dir: Path,
    data_base_dir: Path,
    encoder_name: str,
    seed: int,
    overwrite: bool,
    allow_identity: bool,
) -> list[dict[str, str | int | bool]]:
    source_items = load_items(split, json_path, data_base_dir, source_dir, encoder_name)
    target_items = load_items(split, json_path, data_base_dir, target_dir, encoder_name)
    check_unique_targets(target_items, split)

    missing_source = [item.embedding_path for item in source_items if not item.embedding_path.exists()]
    if missing_source:
        preview = "\n".join(path.as_posix() for path in missing_source[:20])
        raise FileNotFoundError(
            f"Missing {len(missing_source)} source embeddings for split={split}. First missing paths:\n{preview}"
        )

    split_seed = seed + {"train": 0, "val": 1, "test": 2}[split] * 1_000_003
    rng = random.Random(split_seed)
    source_indices = deranged_indices(len(source_items), rng, allow_identity)

    rows: list[dict[str, str | int | bool]] = []
    for target_idx, source_idx in enumerate(source_indices):
        target_item = target_items[target_idx]
        source_item = source_items[source_idx]

        if target_item.embedding_path.resolve(strict=False) == source_item.embedding_path.resolve(strict=False):
            raise RuntimeError(
                "Source and target embedding paths are identical. "
                "Use a different RUN_TAG/target directory for the shuffled control."
            )

        target_item.embedding_path.parent.mkdir(parents=True, exist_ok=True)
        copied = False
        if overwrite or not target_item.embedding_path.exists():
            shutil.copy2(source_item.embedding_path, target_item.embedding_path)
            copied = True

        rows.append(
            {
                "split": split,
                "target_volume": target_item.volume_name,
                "source_volume": source_item.volume_name,
                "target_image_path": target_item.image_path,
                "source_image_path": source_item.image_path,
                "target_embedding_path": target_item.embedding_path.as_posix(),
                "source_embedding_path": source_item.embedding_path.as_posix(),
                "target_index": target_idx,
                "source_index": source_idx,
                "seed": seed,
                "split_seed": split_seed,
                "copied": copied,
            }
        )

    return rows


def write_manifest_csv(rows: list[dict[str, str | int | bool]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "target_volume",
        "source_volume",
        "target_image_path",
        "source_image_path",
        "target_embedding_path",
        "source_embedding_path",
        "target_index",
        "source_index",
        "seed",
        "split_seed",
        "copied",
    ]
    with open(out_csv, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a shuffled text-conditioning control by reassigning real CLIP text embeddings."
    )
    parser.add_argument("--source_embedding_base_dir", required=True)
    parser.add_argument("--target_embedding_base_dir", required=True)
    parser.add_argument("--train_json", default="dataset/unet_train_data_volumes.json")
    parser.add_argument("--val_json", default="dataset/unet_val_data_volumes.json")
    parser.add_argument("--test_json", default="dataset/unet_test_data_volumes.json")
    parser.add_argument("--data_base_dir", default="./data/private_ct")
    parser.add_argument("--report_encoder_model", default="xgem_3D")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_identity",
        action="store_true",
        help="Allow a target volume to keep its own embedding. Disabled by default for a strict negative control.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_embedding_base_dir)
    target_dir = Path(args.target_embedding_base_dir)

    if source_dir.resolve(strict=False) == target_dir.resolve(strict=False):
        raise RuntimeError("source_embedding_base_dir and target_embedding_base_dir must be different.")
    if not source_dir.exists():
        raise FileNotFoundError(f"Source embedding directory does not exist: {source_dir}")

    split_specs = [
        ("train", Path(args.train_json)),
        ("val", Path(args.val_json)),
        ("test", Path(args.test_json)),
    ]

    all_rows: list[dict[str, str | int | bool]] = []
    for split, json_path in split_specs:
        rows = shuffle_split(
            split=split,
            json_path=json_path,
            source_dir=source_dir,
            target_dir=target_dir,
            data_base_dir=Path(args.data_base_dir),
            encoder_name=args.report_encoder_model,
            seed=args.seed,
            overwrite=args.overwrite,
            allow_identity=args.allow_identity,
        )
        all_rows.extend(rows)
        copied = sum(1 for row in rows if row["copied"])
        print(f"[shuffle-emb] split={split} mapped={len(rows)} copied={copied}")

    write_manifest_csv(all_rows, Path(args.manifest_csv))
    print(f"[shuffle-emb] wrote manifest: {args.manifest_csv}")
    print(f"[shuffle-emb] source_dir={source_dir}")
    print(f"[shuffle-emb] target_dir={target_dir}")


if __name__ == "__main__":
    main()
