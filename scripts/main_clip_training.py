import argparse
import json
import os
import sys
from pathlib import Path
import monai

import pandas as pd
import torch
import torch.distributed as dist
from monai.data import DataLoader
from monai.transforms import Compose
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from clip_training.utils.util import mkdir, set_seed, load_config_file
from clip_training.Clip_Training_Script import train
from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model
from clip_training.utils.logger import setup_logger
from scripts.diff_model_setting import distributed_barrier, initialize_distributed

# ===== IMPORT LoRA =====
from lora_layers import apply_lora_to_model

DEFAULT_TRAINER_CONFIG_PATH = ROOT / "clip_training/clip_train_config.yaml"

LEGACY_DATA_PREFIX = "data/private_ct/"
PREPROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "private_ct_preprocessed"
PROCESSED_CT_NAME = "ct_preprocessed.nii.gz"
RAW_CT_NAME = "ct.nii.gz"


def apply_runtime_overrides(config, cli_args):
    overrides = {
        "name": cli_args.override_name,
        "logs": cli_args.override_logs,
        "saved_checkpoints": cli_args.override_saved_checkpoints,
        "num_train_epochs": cli_args.override_num_train_epochs,
        "per_gpu_train_batch_size": cli_args.override_per_gpu_train_batch_size,
        "gradient_accumulation_steps": cli_args.override_gradient_accumulation_steps,
        "logging_steps": cli_args.override_logging_steps,
        "eval_every_epochs": cli_args.override_eval_every_epochs,
        "save_steps_epochs": cli_args.override_save_steps_epochs,
        "num_workers": cli_args.override_num_workers,
        "eval_train_retrieval": cli_args.override_eval_train_retrieval,
        "eval_train_loss": cli_args.override_eval_train_loss,
        "eval_val_retrieval": cli_args.override_eval_val_retrieval,
        "use_amp": cli_args.override_use_amp,
        "max_train_batches_per_epoch": cli_args.override_max_train_batches_per_epoch,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
    return config


def resolve_repo_path(path_str: str | os.PathLike) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    repo_candidate = PROJECT_ROOT / path
    if repo_candidate.exists():
        return repo_candidate

    return path


def resolve_volume_path(image_path: str | os.PathLike) -> Path | None:
    raw_path = Path(image_path)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
        if raw_path.name == PROCESSED_CT_NAME:
            candidates.append(raw_path.with_name(RAW_CT_NAME))
    else:
        candidates.append(PROJECT_ROOT / raw_path)
        candidates.append(Path.cwd() / raw_path)

    normalized = raw_path.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.endswith(PROCESSED_CT_NAME):
        candidates.append(PROJECT_ROOT / normalized.replace(PROCESSED_CT_NAME, RAW_CT_NAME))
    if normalized.startswith(LEGACY_DATA_PREFIX):
        suffix = normalized[len(LEGACY_DATA_PREFIX):]
        candidates.append(PREPROCESSED_DATA_ROOT / suffix)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return None


# def load_filenames(data_list_path: str) -> list:
#     with open(data_list_path, "r") as file:
#         json_data = json.load(file)
#     filenames_train = json_data["training"]
#     return [_item["image"] for _item in filenames_train]

def load_filenames(data_list_path: str, split: str = "training") -> list:
    data_list_path = resolve_repo_path(data_list_path)
    with open(data_list_path, "r") as file:
        json_data = json.load(file)

    if split not in json_data:
        # fallback utile per JSON “monosplit”
        if split == "validation" and "training" in json_data:
            split = "training"
        elif split == "training" and "validation" in json_data:
            split = "validation"
        else:
            raise KeyError(f"Split '{split}' not found in {data_list_path}. Available keys: {list(json_data.keys())}")

    return [_item["image"] for _item in json_data[split]]




def prepare_data(
    train_files: list,
    reports_csv: str,
    config,
    cache_rate: float,
    num_workers: int = 2,
    batch_size: int = 1,
    shuffle: bool = True,
    drop_last: bool = True,
    use_augmentation: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    reports = pd.read_csv(resolve_repo_path(reports_csv))
    volume_text_mapping = {
        row["VolumeName"]: f"Findings: {row['Findings_EN']} Impression: {row['Impressions_EN']}"
        for _, row in reports.iterrows()
    }

    def lookup_text(volume_name: str) -> str:
        return volume_text_mapping.get(volume_name, "")

    # 
    
    # Base transforms (NO preprocessing: volumes already resized/preprocessed)
    transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]

    def _check_shape(x): #still hardcoded 
        if tuple(x.shape) != (1, 512, 512, 128):
            raise ValueError(f"Unexpected volume shape {tuple(x.shape)} (expected [C,512,512,128])")
        return x


    transforms_list.append(monai.transforms.Lambdad(keys="image", func=_check_shape))


    
    # # ===== DATA AUGMENTATION, per ora disattivata =====
    # if use_augmentation and hasattr(config, 'data_augmentation') and config.data_augmentation.get('enabled', False):
    #     aug = config.data_augmentation
    #     transforms_list.append(
    #         monai.transforms.RandAffined(
    #             keys=["image"],
    #             prob=aug.get('probability', 0.5),
    #             rotate_range=(aug.rotation_range, aug.rotation_range, aug.rotation_range),
    #             translate_range=tuple(aug.translation_range),
    #             scale_range=(aug.scale_range, aug.scale_range, aug.scale_range),
    #             mode="bilinear",
    #             padding_mode="border"
    #         )
    #     )
    
    transforms_list.append(monai.transforms.Lambdad(keys="impression", func=lookup_text))
    train_transforms = Compose(transforms_list)

    train_ds = monai.data.CacheDataset(
        data=train_files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers
    )
    persistent_workers = bool(getattr(config, "persistent_workers", False)) and num_workers > 0
    pin_memory = bool(getattr(config, "pin_memory", True))
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return DataLoader(
        train_ds,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
    )


def main():
    parser = argparse.ArgumentParser(description="CLIP3D training entrypoint")
    parser.add_argument(
        "--trainer_config",
        type=str,
        default=str(DEFAULT_TRAINER_CONFIG_PATH),
        help="Path to the CLIP trainer YAML config.",
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Requested GPUs for single-node training.")
    parser.add_argument("--override_name", type=str, default=None)
    parser.add_argument("--override_logs", type=str, default=None)
    parser.add_argument("--override_saved_checkpoints", type=str, default=None)
    parser.add_argument("--override_num_train_epochs", type=int, default=None)
    parser.add_argument("--override_per_gpu_train_batch_size", type=int, default=None)
    parser.add_argument("--override_gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--override_logging_steps", type=int, default=None)
    parser.add_argument("--override_eval_every_epochs", type=int, default=None)
    parser.add_argument("--override_save_steps_epochs", type=int, default=None)
    parser.add_argument("--override_num_workers", type=int, default=None)
    parser.add_argument("--override_eval_train_retrieval", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--override_eval_train_loss", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--override_eval_val_retrieval", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--override_use_amp", type=str, choices=["true", "false"], default=None)
    parser.add_argument("--override_max_train_batches_per_epoch", type=int, default=None)
    parser.add_argument(
        "--finetune_mode",
        type=str,
        choices=["lora", "full"],
        default=None,
        help="Override finetune mode: 'lora' to apply LoRA, 'full' to disable LoRA and do full fine-tuning",
    )
    cli_args = parser.parse_args()

    config = load_config_file(Path(cli_args.trainer_config))
    for attr in (
        "override_eval_train_retrieval",
        "override_eval_train_loss",
        "override_eval_val_retrieval",
        "override_use_amp",
    ):
        value = getattr(cli_args, attr)
        if value is not None:
            setattr(cli_args, attr, value.lower() == "true")
    config = apply_runtime_overrides(config, cli_args)
    # Possibile override della modalità di fine-tuning via CLI (evita hardcoding)
    if getattr(cli_args, "finetune_mode", None) is not None:
        mode = cli_args.finetune_mode
        try:
            if not hasattr(config, "lora") or config.lora is None:
                config.lora = {}
            config.lora["enabled"] = True if mode == "lora" else False
        except Exception:
            # fallback: provare a impostare come attributo semplice
            setattr(config, "lora", {"enabled": True if mode == "lora" else False})
    local_rank, world_size, device = initialize_distributed(cli_args.num_gpus)
    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main_process = rank == 0

    if is_main_process:
        mkdir(path=config.saved_checkpoints)
        mkdir(path=config.logs)
    distributed_barrier(device)

    filename = f"clip_training_logs_{config.name}.txt"
    logger = setup_logger("CLIP TRAINING", config.logs, rank, filename=filename)

    config.device = str(device)
    config.local_rank = local_rank
    config.rank = rank
    config.world_size = world_size
    config.distributed = dist.is_initialized()
    # CLIP reads logit_scale outside the wrapped forward while computing the loss.
    # Static graph lets DDP handle that stable parameter usage without double-ready hooks.
    if getattr(config, "distributed", False) and not hasattr(config, "ddp_find_unused_parameters"):
        config.ddp_find_unused_parameters = False
    if getattr(config, "distributed", False) and not hasattr(config, "ddp_static_graph"):
        config.ddp_static_graph = True
    if getattr(config, "ddp_static_graph", False):
        config.ddp_find_unused_parameters = False
    config.n_gpu = world_size if torch.cuda.is_available() else 0
    set_seed(seed=getattr(config, "seed", 11), n_gpu=config.n_gpu)

    cfgm = model_cfg_bank()(config.clip_model)
    clip = get_model()(cfgm)
    clip = clip.to(config.device)

    # ===== CARICAMENTO CHECKPOINT PRETRAINED =====
    init_ckpt_path = resolve_repo_path(config.init_ckpt) if hasattr(config, "init_ckpt") and config.init_ckpt else None
    if init_ckpt_path and init_ckpt_path.exists():
        ckpt = torch.load(init_ckpt_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = clip.load_state_dict(state, strict=False)
        if is_main_process:
            logger.info(f"Loaded init_ckpt: {init_ckpt_path}")
            logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if len(missing) > 0:
                logger.info(f"Example missing: {missing[0]}")
            if len(unexpected) > 0:
                logger.info(f"Example unexpected: {unexpected[0]}")
    elif is_main_process:
        logger.info("No init_ckpt: training from scratch")

    # ===== APPLICA LoRA =====
    if hasattr(config, 'lora') and config.lora.get('enabled', False):
        if is_main_process:
            logger.info("=" * 60)
            logger.info("APPLYING LoRA ADAPTATION")
            logger.info("=" * 60)
        
        clip = apply_lora_to_model(
            model=clip,
            target_modules=config.lora.target_modules,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            verbose=True
        )

        clip = clip.to(config.device)


        # DOPO apply_lora_to_model
        if is_main_process:
            logger.info("\n=== DIAGNOSTICA PARAMETRI TRAINABILI ===")
            trainable_breakdown = {}
            for name, param in clip.named_parameters():
                if param.requires_grad:
                    module_type = name.split('.')[1] if len(name.split('.')) > 1 else 'root'
                    if module_type not in trainable_breakdown:
                        trainable_breakdown[module_type] = 0
                    trainable_breakdown[module_type] += param.numel()

            for key, count in sorted(trainable_breakdown.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"{key}: {count:,} params")
            logger.info("LoRA applied successfully")
            logger.info("=" * 60)
    else:
        if is_main_process:
            logger.info("LoRA DISABLED - using full fine-tuning (ensuring all params trainable)")
        for _, param in clip.named_parameters():
            param.requires_grad = True
        if is_main_process:
            trainable_params = sum(p.numel() for p in clip.parameters() if p.requires_grad)
            logger.info(f"Trainable params (full-FT): {trainable_params:,}")

    if is_main_process:
        logger.info(f"Training parameters: {config}")

    # ===== PREPARAZIONE DATI =====
    #filenames_train = load_filenames(config.data_list)
    filenames_train = load_filenames(config.data_list, split="training")

    # NUOVO: validation json separato
    filenames_val = load_filenames(config.val_data_list, split="validation")

    train_files = []
    missing_train = []
    for image_path in filenames_train:
        resolved_path = resolve_volume_path(image_path)
        if resolved_path is None:
            missing_train.append(image_path)
            continue
        patient_id = resolved_path.parent.name
        train_files.append({"image": str(resolved_path), "impression": patient_id})
    
    val_files = []
    missing_val = []
    for image_path in filenames_val:
        resolved_path = resolve_volume_path(image_path)
        if resolved_path is None:
            missing_val.append(image_path)
            continue
        patient_id = resolved_path.parent.name
        val_files.append({"image": str(resolved_path), "impression": patient_id})


    if is_main_process:
        logger.info(f"Train files: {len(train_files)}")
        logger.info(f"Val files: {len(val_files)}")
        logger.info(f"Missing train files after path resolution: {len(missing_train)}")
        logger.info(f"Missing val files after path resolution: {len(missing_val)}")
        if missing_train:
            logger.info(f"Example missing train path: {missing_train[0]}")
        if missing_val:
            logger.info(f"Example missing val path: {missing_val[0]}")
    
    # check
    if is_main_process:
        logger.info(f"Example train: {train_files[0] if len(train_files)>0 else 'EMPTY'}")
        logger.info(f"Example val: {val_files[0] if len(val_files)>0 else 'EMPTY'}")

    if not train_files:
        raise FileNotFoundError(
            "No training volumes were found. Checked JSON entries as-is, relative to the repo root, and via "
            f"legacy rewrite '{LEGACY_DATA_PREFIX}' -> '{PREPROCESSED_DATA_ROOT.relative_to(PROJECT_ROOT).as_posix()}/'."
        )
    if not val_files:
        raise FileNotFoundError(
            "No validation volumes were found. Checked JSON entries as-is, relative to the repo root, and via "
            f"legacy rewrite '{LEGACY_DATA_PREFIX}' -> '{PREPROCESSED_DATA_ROOT.relative_to(PROJECT_ROOT).as_posix()}/'."
        )


    # dataloader = prepare_data(
    #     train_files,
    #     reports_csv=config.reports_csv,
    #     config=config,  # pass config per data augmentation
    #     cache_rate=0,
    #     batch_size=config.per_gpu_train_batch_size,
    #     num_workers=config.num_workers,
    # )

    train_loader = prepare_data(
        train_files,
        reports_csv=config.reports_csv,
        config=config,
        cache_rate=0,
        batch_size=config.per_gpu_train_batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
        distributed=config.distributed,
        rank=rank,
        world_size=world_size,
    )

    val_loader = None
    train_eval_loader = None
    val_loss_loader = None
    train_loss_eval_loader = None
    if is_main_process:
        val_loader = prepare_data(
            val_files,
            reports_csv=config.reports_csv,
            config=config,
            cache_rate=0,
            batch_size=1,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            use_augmentation=False,
        )

        train_eval_loader = prepare_data(
            train_files,
            reports_csv=config.reports_csv,
            config=config,
            cache_rate=0,
            batch_size=1,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            use_augmentation=False,
        )

        val_loss_loader = prepare_data(
            val_files,
            reports_csv=config.reports_csv,
            config=config,
            cache_rate=0,
            batch_size=config.per_gpu_train_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            use_augmentation=False,
        )

        train_loss_eval_loader = prepare_data(
            train_files,
            reports_csv=config.reports_csv,
            config=config,
            cache_rate=0,
            batch_size=config.per_gpu_train_batch_size,
            num_workers=config.num_workers,
            shuffle=False,
            drop_last=False,
            use_augmentation=False,
        )




    config.checkpoint_dir = os.path.join(config.saved_checkpoints, config.name)
    if is_main_process:
        mkdir(config.checkpoint_dir)
    distributed_barrier(device)
    
    global_step, avg_loss = train(
        config,
        train_loader,
        clip,
        logger,
        val_retrieval_dataloader=val_loader,
        train_retrieval_dataloader=train_eval_loader,
        val_loss_dataloader=val_loss_loader,
        train_loss_dataloader=train_loss_eval_loader,
    )


    if is_main_process:
        logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
