# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from contextlib import nullcontext
import pandas as pd

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist

import monai
from monai.transforms import Compose
from monai.utils import set_determinism
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from diff_model_setting import initialize_distributed, load_config, setup_logging
from utils import define_instance

# Set the random seed for reproducibility
set_determinism(seed=0)

LEGACY_DATA_PREFIX = "data/private_ct/"
PREPROCESSED_DATA_PREFIX = "data/private_ct_preprocessed/"
PROCESSED_CT_NAME = "ct_preprocessed.nii.gz"
RAW_CT_NAME = "ct.nii.gz"
EXPECTED_LATENT_SHAPE = (128, 128, 32, 4)


def resolve_project_path(path_str: str | os.PathLike) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve(strict=False)


def normalize_prefix(path_str: str | os.PathLike) -> str:
    value = Path(path_str).as_posix()
    if value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def remap_legacy_volume_path(path_str: str | os.PathLike) -> Path:
    path = Path(path_str)
    candidates: list[Path] = []

    def add_variants(candidate: Path, prefer_processed: bool = False) -> None:
        if prefer_processed and candidate.name == RAW_CT_NAME:
            candidates.append(candidate.with_name(PROCESSED_CT_NAME))
            candidates.append(candidate)
        elif candidate.name == PROCESSED_CT_NAME:
            candidates.append(candidate)
            candidates.append(candidate.with_name(RAW_CT_NAME))
        elif candidate.name == RAW_CT_NAME:
            candidates.append(candidate)
            candidates.append(candidate.with_name(PROCESSED_CT_NAME))
        else:
            candidates.append(candidate)

    if path.is_absolute():
        absolute_posix = path.as_posix()
        legacy_marker = f"/{LEGACY_DATA_PREFIX}"
        if legacy_marker in absolute_posix:
            remapped = absolute_posix.replace(legacy_marker, f"/{PREPROCESSED_DATA_PREFIX}", 1)
            add_variants(Path(remapped), prefer_processed=True)
        add_variants(path)
    else:
        normalized = normalize_prefix(path)
        if normalized.startswith(LEGACY_DATA_PREFIX):
            suffix = normalized[len(LEGACY_DATA_PREFIX):]
            add_variants(resolve_project_path(PREPROCESSED_DATA_PREFIX + suffix), prefer_processed=True)
        add_variants(resolve_project_path(path))

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else resolve_project_path(path)


def resolve_input_volume_path(filepath: str | os.PathLike, data_base_dir: str | os.PathLike) -> Path:
    if os.path.isabs(filepath):
        raw_input_path = Path(filepath)
    else:
        data_base_dir = normalize_prefix(data_base_dir)
        normalized_fp = normalize_prefix(filepath)
        if (
            normalized_fp.startswith(data_base_dir)
            or normalized_fp.startswith(PREPROCESSED_DATA_PREFIX)
            or normalized_fp.startswith(LEGACY_DATA_PREFIX)
        ):
            raw_input_path = resolve_project_path(normalized_fp)
        else:
            raw_input_path = resolve_project_path(f"{data_base_dir}/{normalized_fp}")

    return remap_legacy_volume_path(raw_input_path)


def canonical_latent_rel_path(rel_path: str) -> str:
    if rel_path.endswith(PROCESSED_CT_NAME):
        return rel_path[: -len(PROCESSED_CT_NAME)] + RAW_CT_NAME
    return rel_path


def latent_output_path(filepath: str | os.PathLike, embedding_base_dir: str | os.PathLike, data_base_dir: str | os.PathLike) -> Path:
    embedding_base_dir = resolve_project_path(embedding_base_dir)
    data_base_dir = normalize_prefix(data_base_dir)

    if os.path.isabs(filepath):
        input_posix = Path(filepath).as_posix()
    else:
        normalized_fp = normalize_prefix(filepath)
        if (
            normalized_fp.startswith(data_base_dir)
            or normalized_fp.startswith(PREPROCESSED_DATA_PREFIX)
            or normalized_fp.startswith(LEGACY_DATA_PREFIX)
        ):
            input_posix = normalized_fp
        else:
            input_posix = f"{data_base_dir}/{normalized_fp}"

    rel_path = None
    for prefix in (data_base_dir, LEGACY_DATA_PREFIX.rstrip("/"), PREPROCESSED_DATA_PREFIX.rstrip("/")):
        marker = f"/{prefix}/"
        if marker in input_posix:
            rel_path = input_posix.split(marker, 1)[1]
            break
        if input_posix.startswith(prefix + "/"):
            rel_path = input_posix[len(prefix) + 1:]
            break

    if rel_path is None:
        raise ValueError(f"Cannot derive relative latent path from input: {filepath}")

    return embedding_base_dir / canonical_latent_rel_path(rel_path)


def latent_has_expected_shape(path: Path, expected_shape: tuple[int, ...] = EXPECTED_LATENT_SHAPE) -> bool:
    try:
        return tuple(nib.load(str(path), mmap=False).shape) == tuple(expected_shape)
    except Exception:
        return False


### cambio la funzione
def create_transforms(dim: tuple = None, device="cuda") -> Compose:
    # dim/device sono lasciati per compatibilità, ma NON usati
    return Compose(
        [
            monai.transforms.LoadImaged(keys="image"),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys="image", dtype=torch.float32),
        ]
    )




def round_number(number: int, base_number: int = 128) -> int:
    """
    Round the number to the nearest multiple of the base number, with a minimum value of the base number.

    Args:
        number (int): Number to be rounded.
        base_number (int): Number to be common divisor.

    Returns:
        int: Rounded number.
    """
    new_number = max(round(float(number) / float(base_number)), 1.0) * float(base_number)
    return int(new_number)


def load_filenames(data_list_path: str, split: str = "training") -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(resolve_project_path(data_list_path), "r") as file:
        json_data = json.load(file)
    if split not in json_data:
        available = ", ".join(sorted(json_data.keys()))
        raise KeyError(f"Split '{split}' not found in {data_list_path}. Available keys: {available}")

    filenames_raw = json_data[split]
    return [_item["image"] for _item in filenames_raw]


def save_filenames(data_list_path: str, filenames_raw: list):
    """
    Save the updated filenames back to the JSON file. If the file doesn't exist, it will be created.

    Args:
        data_list_path (str): Path to the JSON data list file.
        filenames_raw (list): List of updated filenames.
    """
    # Check if the file exists, if not create an empty structure
    if not os.path.exists(data_list_path):
        json_data = {"training": []}
    else:
        with open(data_list_path, "r") as file:
            json_data = json.load(file)
    
    # Rebuild the original JSON structure
    json_data["training"] = [{"image": filename} for filename in filenames_raw]

    # Save the updated JSON data
    with open(data_list_path, "w") as file:
        json.dump(json_data, file, indent=4)

def process_file(
    filepath: str,
    args: argparse.Namespace,
    autoencoder: torch.nn.Module,
    device: torch.device,
    plain_transforms: Compose,
    new_transforms: Compose,
    logger: logging.Logger,
    overwrite_bad_shape: bool = False,
) -> None:
    """
    Process a single file to create training latents (.nii.gz) with VAE.
    """

    # 1) costruisci input e output path
    out_filename = latent_output_path(filepath, args.embedding_base_dir, args.data_base_dir)
    input_path = resolve_input_volume_path(filepath, args.data_base_dir)

    # 2) se già esiste -> skip
    if out_filename.is_file():
        if overwrite_bad_shape and not latent_has_expected_shape(out_filename):
            logger.warning("Regenerating latent with unexpected shape: %s", out_filename)
        else:
            print("Already_done:", out_filename)
            return

    # 3) definisci nome temporaneo con ESTENSIONE VALIDA (fondamentale!)
    #    così nibabel non fallisce sul save
    tmp_name = str(out_filename).replace(".nii.gz", ".tmp.nii.gz")

    # 4) se era rimasto un tmp da run precedente, eliminalo
    if os.path.isfile(tmp_name):
        try:
            os.remove(tmp_name)
        except Exception:
            pass

    # 5) carica volume (plain) solo per leggere meta/dim (debug)
    test_data = {"image": str(input_path)}
    logger.info(f"input_path: {input_path}")
    # transformed_data = plain_transforms(test_data)
    # nda = transformed_data["image"]

    # dim = [int(nda.meta["dim"][_i]) for _i in range(1, 4)]
    # spacing = [0.75, 0.75, 3.0]
    # logger.info(f"old dim: {dim}, old spacing: {spacing}")

    # 6) carica volume già preprocessato (no resize/no intensity scaling)
    new_data = new_transforms(test_data)
    nda_image = new_data["image"]  # shape tipica: (C,H,W,D)

    # affine per salvare nifti
    new_affine = nda_image.meta["affine"].cpu().numpy()
    logger.info(f"new dim: {nda_image.shape}, new affine: {new_affine}")

    # crea cartella destinazione
    out_filename.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"out_filename: {out_filename}")

    # 7) prepara input per VAE: (B,C,H,W,D)
    #    nda_image è già torch.Tensor (EnsureTyped), su CPU
    pt_nda = nda_image.float()
    if pt_nda.ndim == 4:        # (C,H,W,D)
        pt_nda = pt_nda.unsqueeze(0)  # -> (B,C,H,W,D)

    # 8) encode su GPU
    autocast_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

    with torch.inference_mode():
        with autocast_ctx:
            pt_nda = pt_nda.to(device, non_blocking=True)
            z = autoencoder.encode_stage_2_inputs(pt_nda)
            logger.info(f"z: {tuple(z.shape)}, {z.dtype}")

    # 9) prepara output nifti: z[0] è (4,128,128,32) -> (128,128,32,4)
    out_nda = z[0].detach().float().cpu().numpy().transpose(1, 2, 3, 0)
    out_img = nib.Nifti1Image(np.float32(out_nda), affine=new_affine)

    # 10) salva prima in tmp (estensione valida), poi rename atomico
    nib.save(out_img, tmp_name)
    os.replace(tmp_name, out_filename)



@torch.inference_mode()
def diff_model_create_training_data(
    env_config_path: str,
    model_config_path: str,
    model_def_path: str,
    num_gpus: int,
    index: int,
    split: str = "training",
    json_data_list_override: str | None = None,
    embedding_base_dir_override: str | None = None,
    data_base_dir_override: str | None = None,
    chunk_size: int = 500,
    only_missing: bool = False,
    overwrite_bad_shape: bool = False,
) -> None:
    """
    Create training data for the diffusion model.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    if embedding_base_dir_override:
        args.embedding_base_dir = embedding_base_dir_override
    if data_base_dir_override:
        args.data_base_dir = data_base_dir_override
    local_rank, world_size, device = initialize_distributed(num_gpus=num_gpus)
    logger = setup_logging("creating training data")
    logger.info(f"Using device {device}")

    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")
    autoencoder.eval()

    embedding_base_dir = resolve_project_path(args.embedding_base_dir)
    embedding_base_dir.mkdir(parents=True, exist_ok=True)

    json_data_list = json_data_list_override if json_data_list_override else args.json_data_list
    filenames_raw = load_filenames(json_data_list, split=split)
    if chunk_size > 0:
        filenames_raw = filenames_raw[index:index + chunk_size]
    else:
        filenames_raw = filenames_raw[index:]

    if only_missing:
        missing_files = []
        for filepath in filenames_raw:
            out_filename = latent_output_path(filepath, args.embedding_base_dir, args.data_base_dir)
            if not out_filename.is_file() or (overwrite_bad_shape and not latent_has_expected_shape(out_filename)):
                missing_files.append(filepath)
        filenames_raw = missing_files

    logger.info(
        "Preparing latent generation: split=%s, json=%s, start_index=%d, chunk_size=%d, selected_files=%d, only_missing=%s, overwrite_bad_shape=%s",
        split,
        json_data_list,
        index,
        chunk_size,
        len(filenames_raw),
        only_missing,
        overwrite_bad_shape,
    )

    plain_transforms = create_transforms(dim=None)

    error_log_path = embedding_base_dir / "error_paths.txt"
    if error_log_path.exists():
        error_log_path.unlink()

    #new_dim = (512, 512, 128)
    new_transforms = plain_transforms

    for _iter in tqdm(range(len(filenames_raw))):
        if _iter % world_size != local_rank:
            continue

        filepath = filenames_raw[_iter]

        try:
            #new_dim = (512, 512, 128)
            #new_transforms = create_transforms(new_dim)

            process_file(
                filepath,
                args,
                autoencoder,
                device,
                plain_transforms,
                new_transforms,
                logger,
                overwrite_bad_shape=overwrite_bad_shape,
            )
        except Exception as e:
            is_cuda_oom = isinstance(e, torch.OutOfMemoryError) or "CUDA out of memory" in str(e)
            if is_cuda_oom and device.type == "cuda":
                logger.warning("CUDA OOM on %s, skipping this volume", filepath)
            else:
                print(filepath)
                logger.exception("Failed to create latent for %s", filepath)
            error_path = str(remap_legacy_volume_path(filepath))
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log_path, "a") as f:
                f.write(f"{error_path}\n")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training Data Creation")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def", type=str, default="./configs/config_rflow.json", help="Path to model definition file"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for distributed training")
    parser.add_argument("--index", type=int, default=0, help="Index of the batch to process")
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "validation", "test"],
        help="JSON split key to process",
    )
    parser.add_argument(
        "--json_data_list_override",
        type=str,
        default=None,
        help="Optional JSON datalist to override env_config json_data_list",
    )
    parser.add_argument(
        "--embedding_base_dir_override",
        type=str,
        default=None,
        help="Optional output root for VAE latents, overriding env_config embedding_base_dir",
    )
    parser.add_argument(
        "--data_base_dir_override",
        type=str,
        default=None,
        help="Optional input root for source volumes, overriding env_config data_base_dir",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Number of files to process from index (<=0 means process all remaining files)",
    )
    parser.add_argument(
        "--only_missing",
        action="store_true",
        help="Process only files whose latent output does not exist yet",
    )
    parser.add_argument(
        "--overwrite_bad_shape",
        action="store_true",
        help=f"Regenerate existing latent files whose shape is not {EXPECTED_LATENT_SHAPE}",
    )


    args = parser.parse_args()
    diff_model_create_training_data(
        args.env_config,
        args.model_config,
        args.model_def,
        args.num_gpus,
        args.index,
        split=args.split,
        json_data_list_override=args.json_data_list_override,
        embedding_base_dir_override=args.embedding_base_dir_override,
        data_base_dir_override=args.data_base_dir_override,
        chunk_size=args.chunk_size,
        only_missing=args.only_missing,
        overwrite_bad_shape=args.overwrite_bad_shape,
    )
