"""
Encode CT-RATE reports into text embeddings using the CLIP3D checkpoint.

- Reads volume lists from the train/validation JSON files (relative paths).
- Looks up corresponding reports from CSVs (columns: VolumeName, Findings_EN, Impressions_EN).
- Saves embeddings under the embedding base directory, matching the convention
  expected by the diffusion pipeline: <embedding_base>/<volume>_impression_<report_encoder_model>.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model
from scripts.diff_model_setting import initialize_distributed


LEGACY_DATA_PREFIX = "data/private_ct"
PREPROCESSED_DATA_PREFIX = "data/private_ct_preprocessed"
PROCESSED_CT_NAME = "ct_preprocessed.nii.gz"
RAW_CT_NAME = "ct.nii.gz"


def load_volume_list(json_path: Path) -> list[str]:
    with open(json_path, "r") as file:
        data = json.load(file)
    key = "training" if "training" in data else ("validation" if "validation" in data else "test")
    return [_item["image"] for _item in data.get(key, [])]


def maybe_limit_items(items: list[str], max_items: int) -> list[str]:
    if max_items <= 0:
        return items
    return items[:max_items]


def strip_nii_gz(path: str) -> str:
    return path[:-7] if path.endswith(".nii.gz") else Path(path).with_suffix("").as_posix()


def build_embedding_path(image_path: str, data_base_dir: Path, embedding_base_dir: Path, encoder_name: str) -> Path:
    img_path = Path(image_path)
    rel_str = img_path.as_posix().lstrip("./")
    if rel_str.endswith(PROCESSED_CT_NAME):
        rel_str = rel_str.replace(PROCESSED_CT_NAME, RAW_CT_NAME)

    base_prefixes = [
        data_base_dir.as_posix().lstrip("./").rstrip("/"),
        LEGACY_DATA_PREFIX,
        PREPROCESSED_DATA_PREFIX,
    ]
    for base_prefix in dict.fromkeys(base_prefixes):
        if base_prefix and rel_str.startswith(base_prefix + "/"):
            rel_str = rel_str[len(base_prefix) + 1:]
            break

    base = strip_nii_gz(rel_str)
    return embedding_base_dir / f"{base}_impression_{encoder_name}.npy"


def shard_items(items: list[str], rank: int, world_size: int) -> list[str]:
    if world_size <= 1:
        return items
    return [item for idx, item in enumerate(items) if (idx % world_size) == rank]


def flush_batch(clip, texts: list[str], output_paths: list[Path]) -> int:
    if not texts:
        return 0

    with torch.inference_mode():
        embeddings = clip(texts, "encode_text").cpu().numpy().astype(np.float32)

    for emb_path, embedding in zip(output_paths, embeddings):
        np.save(emb_path, embedding)
    return len(output_paths)


def main(args: argparse.Namespace) -> None:
    data_base_dir = Path(args.data_base_dir)
    embedding_base_dir = Path(args.embedding_base_dir)
    encoder_name = args.report_encoder_model
    local_rank, world_size, device = initialize_distributed(args.num_gpus)
    rank = dist.get_rank() if dist.is_initialized() else 0

    train_images = maybe_limit_items(load_volume_list(Path(args.train_json)), args.max_items_per_split)
    val_images = maybe_limit_items(load_volume_list(Path(args.val_json)), args.max_items_per_split)

    reports_train = pd.read_csv(args.train_reports).set_index("VolumeName")
    reports_val = pd.read_csv(args.val_reports).set_index("VolumeName")

    cfgm = model_cfg_bank()("clip_3D")
    clip = get_model()(cfgm)
    clip_weights = Path(args.clip_weights)
    ckpt = torch.load(clip_weights, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = clip.load_state_dict(state, strict=False)
    if rank == 0:
        print(f"[CLIP LOAD] missing={len(missing)} unexpected={len(unexpected)}")

    clip.to(device)
    clip.eval()

    def encode_impressions(image_list: list[str], reports_df: pd.DataFrame):
        local_images = shard_items(image_list, rank, world_size)
        if rank == 0:
            print(
                f"[EMB] total_images={len(image_list)} world_size={world_size} "
                f"local_rank={local_rank} local_images={len(local_images)} batch_size={args.batch_size}"
            )

        pending_texts: list[str] = []
        pending_paths: list[Path] = []
        saved = 0
        skipped_existing = 0
        skipped_missing_report = 0

        iterator = tqdm(local_images, desc="encoding reports", position=local_rank, disable=False)
        for img in iterator:
            volume_name = Path(img).parent.name

            if volume_name not in reports_df.index:
                skipped_missing_report += 1
                continue
            findings = str(reports_df.loc[volume_name, "Findings_EN"])
            impressions = str(reports_df.loc[volume_name, "Impressions_EN"])
            text = f"Findings: {findings} Impression: {impressions}"

            emb_path = build_embedding_path(img, data_base_dir, embedding_base_dir, encoder_name)
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            if emb_path.exists() and not args.overwrite:
                skipped_existing += 1
                continue

            pending_texts.append(text)
            pending_paths.append(emb_path)

            if len(pending_texts) >= args.batch_size:
                saved += flush_batch(clip, pending_texts, pending_paths)
                pending_texts = []
                pending_paths = []

        saved += flush_batch(clip, pending_texts, pending_paths)
        print(
            f"[EMB][rank={rank}] saved={saved} skipped_existing={skipped_existing} "
            f"skipped_missing_report={skipped_missing_report}"
        )

    if args.test_json and args.test_reports:
        test_images = maybe_limit_items(load_volume_list(Path(args.test_json)), args.max_items_per_split)
        reports_test = pd.read_csv(args.test_reports).set_index("VolumeName")
        encode_impressions(test_images, reports_test)

    encode_impressions(train_images, reports_train)
    encode_impressions(val_images, reports_val)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode CT-RATE reports to embeddings")
    parser.add_argument("--train_json", type=str, default="data/train_data_volumes.json")
    parser.add_argument("--val_json", type=str, default="data/validation_data_volumes.json")
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--train_reports", type=str, default="data/train_reports.csv")
    parser.add_argument("--val_reports", type=str, default="data/validation_reports.csv")
    parser.add_argument("--test_reports", type=str, default=None)
    parser.add_argument("--data_base_dir", type=str, default="dataset")
    parser.add_argument("--embedding_base_dir", type=str, default="./embeddings")
    parser.add_argument("--clip_weights", type=str, default="./models/CLIP3D_Finding_Impression_30ep.pt")
    parser.add_argument("--report_encoder_model", type=str, default="xgem_3D")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_items_per_split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing embeddings")
    args = parser.parse_args()
    main(args)
