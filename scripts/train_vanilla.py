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
from datetime import datetime
from pathlib import Path
import numpy as np

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import monai
from monai.data import DataLoader
from monai.transforms import Compose
from monai.utils import first
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.utils import define_instance
import warnings
warnings.filterwarnings("ignore")


def _load_json_field(path, field):
    with open(path, "r") as f:
        data = json.load(f)
    return data.get(field, "")


def _load_npy(path):
    return torch.tensor(np.load(path), dtype=torch.float32)


def load_filenames(data_list_path: str, split: str = "training") -> list:
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
    return [item["image"] for item in json_data[split]]


def prepare_data(
    train_files: list, device: torch.device, cache_rate: float,
    num_workers: int = 2, batch_size: int = 1,
    include_body_region: bool = False, shuffle: bool = True, drop_last: bool = True,
    distributed: bool = False, rank: int = 0, world_size: int = 1,
) -> DataLoader:

    def _load_data_from_file(file_path, key):
        with open(file_path) as f:
            return torch.FloatTensor(json.load(f)[key])

    transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_data_from_file(x, "spacing")),
        monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
        monai.transforms.Lambdad(keys="cond", func=lambda x: _load_npy(x)),
        monai.transforms.Lambdad(keys="impression", func=lambda x: _load_json_field(x, "impression")),
    ]

    if include_body_region:
        transforms_list += [
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: _load_data_from_file(x, "top_region_index")),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: _load_data_from_file(x, "bottom_region_index")),
            monai.transforms.Lambdad(keys="top_region_index", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="bottom_region_index", func=lambda x: x * 1e2),
        ]

    ds = monai.data.CacheDataset(
        data=train_files, transform=Compose(transforms_list),
        cache_rate=cache_rate, num_workers=num_workers
    )
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return DataLoader(
        ds,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
    )


def load_unet(args, device, logger):
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if dist.is_initialized():
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        ddp_output_device = device.index if device.type == "cuda" else None
        unet = DistributedDataParallel(
            unet,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            find_unused_parameters=False,
        )

    if args.existing_ckpt_filepath is None:
        logger.info("CORRETTO")
        logger.info("Training from scratch.")
    else:
        ckpt = torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)
        sd = ckpt.get("unet_state_dict", ckpt)
        if dist.is_initialized():
            unet.module.load_state_dict(sd, strict=False)
        else:
            unet.load_state_dict(sd, strict=False)
        logger.info(f"Pretrained checkpoint {args.existing_ckpt_filepath} loaded.")

    return unet


def load_training_state(args, device, logger, unet, optimizer, scheduler, scaler):
    ckpt = torch.load(args.existing_ckpt_filepath, map_location=device, weights_only=False)
    has_full_state = all(key in ckpt for key in ("optimizer", "scheduler", "grad_scaler"))
    if has_full_state:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler = ckpt["scheduler"]
        scaler = ckpt["grad_scaler"]
        logger.info("Training state loaded (optimizer, scheduler, scaler).")
    else:
        logger.warning(
            "Checkpoint %s does not contain optimizer/scheduler/scaler; resuming with fresh optimizer state.",
            args.existing_ckpt_filepath,
        )
        logger.info("Training weights loaded; optimizer state kept fresh.")
    return optimizer, scheduler, scaler


def calculate_scale_factor(train_loader, device, logger):
    z = first(train_loader)["image"].to(device)
    scale_factor = 1 / torch.std(z)
    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor


def create_optimizer(model, lr):
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=2.0)


def _log_missing_examples(logger, split_name, category, examples, limit=5):
    for idx, example in enumerate(examples[:limit], 1):
        logger.warning("[%s] missing %s example %d: %s", split_name, category, idx, example)


def _run_one_epoch(
    epoch, unet, loader, noise_scheduler, loss_pt, scale_factor,
    num_train_timesteps, device, logger, local_rank, amp,
    freq_to_print, conditional_free_guidance,
    ct_property_conditions, include_modality,
    optimizer=None, lr_scheduler=None, scaler=None, is_train=True
):
    tag = "TRAIN" if is_train else "VAL"
    if is_train:
        unet.train()
    else:
        unet.eval()

    loss_torch = torch.zeros(2, dtype=torch.float, device=device)

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for _iter, data in enumerate(loader, 1):
            images = data["image"].to(device) * scale_factor
            spacing = data["spacing"].to(device)
            cond = data["cond"].to(device)
            if cond.ndim == 4:
                cond = cond.squeeze(1)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                noise = torch.randn_like(images)
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(images)
                else:
                    timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=device).long()

                noisy = noise_scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)

                mask = torch.rand(images.shape[0], device=device) < conditional_free_guidance
                cond_masked = cond.clone()
                cond_masked[mask] = 0

                inputs = {"x": noisy, "timesteps": timesteps, "spacing_tensor": spacing, "context": cond_masked}

                if ct_property_conditions:
                    inputs["top_region_index_tensor"] = data["top_region_index"].to(device)
                    inputs["bottom_region_index_tensor"] = data["bottom_region_index"].to(device)
                if include_modality:
                    inputs["class_labels"] = torch.ones(len(images), dtype=torch.long, device=device)

                out = unet(**inputs)

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    gt = images
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    gt = images - noise
                else:
                    raise ValueError(f"Invalid prediction type: {noise_scheduler.prediction_type}")

                loss = loss_pt(out.float(), gt.float())

            if is_train:
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                lr_scheduler.step()

            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0

            if local_rank == 0 and (_iter % freq_to_print == 0 or _iter == 1):
                lr = optimizer.param_groups[0]["lr"] if is_train else 0
                logger.info(
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}][{tag}] epoch {epoch+1}, "
                    f"iter {_iter}/{len(loader)}, loss: {loss.item():.4f}" +
                    (f", lr: {lr:.12f}" if is_train else "")
                )

    if dist.is_initialized():
        dist.all_reduce(loss_torch, op=torch.distributed.ReduceOp.SUM)

    return loss_torch


def save_checkpoint(
    epoch, unet, train_loss, num_train_timesteps, scale_factor,
    ckpt_folder, args, optimizer=None, scheduler=None, grad_scaler=None, val_loss=None
):
    sd = unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    payload = {
        "epoch": epoch + 1,
        "num_train_timesteps": num_train_timesteps,
        "scale_factor": scale_factor,
        "unet_state_dict": sd,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    if optimizer is not None:
        payload.update({
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler,
            "grad_scaler": grad_scaler,
        })
        suffix = f"_{epoch+1}_resume.pt"
    else:
        suffix = f"_{epoch+1}.pt"

    path = f"{ckpt_folder}/{str(args.model_filename).replace('.pt', suffix)}"
    torch.save(payload, path)


def build_file_dicts(filenames, args, include_body_region, logger=None, split_name="train"):
    def _norm(path_str: str) -> str:
        s = Path(path_str).as_posix()
        if s.startswith("./"):
            s = s[2:]
        return s.rstrip("/")

    legacy_prefix = "data/private_ct/"
    preproc_prefix = "data/private_ct_preprocessed/"
    processed_ct_name = "ct_preprocessed.nii.gz"
    raw_ct_name = "ct.nii.gz"
    data_base_norm = _norm(args.data_base_dir)
    emb_base = Path(args.embedding_base_dir)
    cond_base = Path(getattr(args, "conditioning_base_dir", args.embedding_base_dir))

    def _resolve_image_path(raw: str) -> Path:
        p = Path(raw)
        candidates: list[Path] = []

        if p.is_absolute():
            candidates.append(p)
            if p.name == processed_ct_name:
                candidates.append(p.with_name(raw_ct_name))
        else:
            raw_norm = _norm(raw)
            # Try project-relative path first (works when JSON already stores data/...)
            candidates.append((ROOT / raw_norm).resolve(strict=False))
            if raw_norm.endswith(processed_ct_name):
                candidates.append((ROOT / raw_norm.replace(processed_ct_name, raw_ct_name)).resolve(strict=False))
            # Then try data_base_dir + relative path (works for bare relative entries)
            candidates.append((ROOT / data_base_norm / raw_norm).resolve(strict=False))

            # Legacy mapping: data/private_ct/... -> data/private_ct_preprocessed/...
            if raw_norm.startswith(legacy_prefix):
                remapped = preproc_prefix + raw_norm[len(legacy_prefix):]
                candidates.append((ROOT / remapped).resolve(strict=False))

        seen = set()
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            if c.exists():
                return c

        return candidates[-1]

    def _relative_volume_path(img_path: Path) -> str | None:
        posix = img_path.as_posix()
        for marker in (f"/{preproc_prefix}", f"/{legacy_prefix}"):
            if marker in posix:
                return posix.split(marker, 1)[1]
        for prefix in (preproc_prefix.rstrip("/"), legacy_prefix.rstrip("/")):
            if posix.startswith(prefix + "/"):
                return posix[len(prefix) + 1:]
        return None

    files = []
    missing_source = []
    missing_rel_path = []
    missing_latent = []
    missing_sidecar = []
    missing_cond = []

    for f in filenames:
        img_path = _resolve_image_path(f)
        if not img_path.exists():
            missing_source.append(str(img_path))
            continue

        rel_path = _relative_volume_path(img_path)
        if rel_path is None:
            missing_rel_path.append(str(img_path))
            continue

        emb_path = emb_base / rel_path
        info_path = Path(str(emb_path) + ".json")
        info_path = Path(str(info_path).replace("_emb", ""))
        cond_path = cond_base / rel_path
        cond_path = Path(str(cond_path).replace(".nii.gz", f"_impression_{args.report_encoder_model}.npy"))

        sample_missing = False
        if not emb_path.exists():
            missing_latent.append(str(emb_path))
            sample_missing = True
        if not info_path.exists():
            missing_sidecar.append(str(info_path))
            sample_missing = True
        if not cond_path.exists():
            missing_cond.append(str(cond_path))
            sample_missing = True
        if sample_missing:
            continue

        d = {"image": emb_path.as_posix(), "spacing": info_path.as_posix(), "impression": info_path.as_posix(), "cond": cond_path.as_posix()}
        if include_body_region:
            d["top_region_index"] = info_path.as_posix()
            d["bottom_region_index"] = info_path.as_posix()
        files.append(d)

    if logger is not None:
        skipped = len(filenames) - len(files)
        logger.info(
            "[%s] usable samples=%d/%d, skipped=%d (source_missing=%d, rel_path_missing=%d, latent_missing=%d, sidecar_missing=%d, cond_missing=%d)",
            split_name,
            len(files),
            len(filenames),
            skipped,
            len(missing_source),
            len(missing_rel_path),
            len(missing_latent),
            len(missing_sidecar),
            len(missing_cond),
        )
        _log_missing_examples(logger, split_name, "source", missing_source)
        _log_missing_examples(logger, split_name, "relative-path", missing_rel_path)
        _log_missing_examples(logger, split_name, "latent", missing_latent)
        _log_missing_examples(logger, split_name, "sidecar", missing_sidecar)
        _log_missing_examples(logger, split_name, "conditioning", missing_cond)

    return files


def apply_runtime_overrides(args, runtime_args):
    if runtime_args is None:
        return args

    if runtime_args.override_batch_size is not None:
        args.diffusion_unet_train["batch_size"] = runtime_args.override_batch_size
    if runtime_args.override_n_epochs is not None:
        args.diffusion_unet_train["n_epochs"] = runtime_args.override_n_epochs
    if runtime_args.override_n_epochs_total is not None:
        args.diffusion_unet_train["n_epochs_total"] = runtime_args.override_n_epochs_total
    if runtime_args.override_freq_to_print is not None:
        args.diffusion_unet_train["freq_to_print"] = runtime_args.override_freq_to_print
    if runtime_args.override_save_epoch_freq is not None:
        args.diffusion_unet_train["save_epoch_freq"] = runtime_args.override_save_epoch_freq
    if runtime_args.override_model_dir is not None:
        args.model_dir = runtime_args.override_model_dir
    if runtime_args.override_model_filename is not None:
        args.model_filename = runtime_args.override_model_filename
    if runtime_args.override_embedding_base_dir is not None:
        args.embedding_base_dir = runtime_args.override_embedding_base_dir
    if runtime_args.override_conditioning_base_dir is not None:
        args.conditioning_base_dir = runtime_args.override_conditioning_base_dir
    if runtime_args.override_log_dir is not None:
        args.log_dir = runtime_args.override_log_dir
    if runtime_args.override_existing_ckpt_filepath is not None:
        args.existing_ckpt_filepath = runtime_args.override_existing_ckpt_filepath
    if runtime_args.override_continue_training_from is not None:
        args.diffusion_unet_train["continue_training_from"] = runtime_args.override_continue_training_from

    return args


def diff_model_train(env_config_path, model_config_path, model_def_path, num_gpus, amp=True, runtime_args=None):
    args = load_config(env_config_path, model_config_path, model_def_path)
    args = apply_runtime_overrides(args, runtime_args)
    if not getattr(args, "conditioning_base_dir", None):
        args.conditioning_base_dir = args.embedding_base_dir
    

    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")
    

    # --- File logger ---
    if local_rank == 0:
        model_dir_p = Path(args.model_dir).resolve()
        log_dir = Path(getattr(args, "log_dir", "")) if getattr(args, "log_dir", None) else ROOT / "logs" / model_dir_p.parent.name / model_dir_p.name
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("[%(asctime)s.%(msecs)03d][%(levelname)s](%(name)s) - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info(f"Using {device} of {world_size}")
        logger.info(f"[config] ckpt_folder={args.model_dir} lr={args.diffusion_unet_train['lr']} "
                     f"n_epochs={args.diffusion_unet_train['n_epochs']} "
                     f"num_train_timesteps={args.noise_scheduler['num_train_timesteps']}")
        logger.info(f"[config] latent_root={args.embedding_base_dir}")
        logger.info(f"[config] conditioning_root={args.conditioning_base_dir}")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"continue_training_from = {args.diffusion_unet_train.get('continue_training_from')}")
    logger.info(f"existing_ckpt_filepath  = {args.existing_ckpt_filepath}")

    # --- Data ---
    filenames_train = load_filenames(args.json_data_list)
    filenames_val = load_filenames(args.val_json_data_list, split="validation")
    if local_rank == 0:
        logger.info(f"num_files_train={len(filenames_train)}, num_files_val={len(filenames_val)}")

    unet = load_unet(args, device, logger)

    # === DEBUG: count trainable params (put HERE) ===
    if (not dist.is_initialized()) or local_rank == 0:
        m = unet.module if dist.is_initialized() else unet
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        logger.info(f"[DEBUG] params trainable={trainable} / total={total}  (ratio={trainable/total:.6f})")

    noise_scheduler = define_instance(args, "noise_scheduler")

    mod = unet.module if dist.is_initialized() else unet
    include_body_region = mod.include_top_region_index_input
    include_modality = mod.num_class_embeds is not None

    train_files = build_file_dicts(filenames_train, args, include_body_region, logger=logger, split_name="train")
    val_files = build_file_dicts(filenames_val, args, include_body_region, logger=logger, split_name="validation")

    if not train_files:
        raise RuntimeError("No usable training samples after artifact validation.")
    if filenames_val and not val_files:
        raise RuntimeError("No usable validation samples after artifact validation.")

    train_loader = prepare_data(train_files, device, args.diffusion_unet_train["cache_rate"],
                                batch_size=args.diffusion_unet_train["batch_size"], include_body_region=include_body_region,
                                distributed=dist.is_initialized(), rank=local_rank, world_size=world_size)
    val_loader = prepare_data(val_files, device, args.diffusion_unet_train["cache_rate"],
                              batch_size=args.diffusion_unet_train["batch_size"], include_body_region=include_body_region,
                              shuffle=False, drop_last=False,
                              distributed=dist.is_initialized(), rank=local_rank, world_size=world_size)

    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])
    # === DEBUG: count optimizer params (put HERE) ===
    if (not dist.is_initialized()) or local_rank == 0:
        opt_params = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
        logger.info(f"[DEBUG] optimizer params={opt_params}")
    total_steps = args.diffusion_unet_train["n_epochs_total"] * len(train_loader)
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler()
    torch.set_float32_matmul_precision("highest")

    if args.diffusion_unet_train.get("continue_training_from") is not None:
        optimizer, lr_scheduler, scaler = load_training_state(args, device, logger, unet, optimizer, lr_scheduler, scaler)
        start = args.diffusion_unet_train["continue_training_from"]
        epochs = range(start, start + args.diffusion_unet_train["n_epochs"])
    else:
        logger.info(f"SONO ENTRATO CORRETTAMENTE")
        epochs = range(args.diffusion_unet_train["n_epochs"])


    val_cfg_prob = float(getattr(args, "val_cfg_probability", 0.0))
    save_every = int(args.diffusion_unet_train.get("save_epoch_freq", 50))
    last_epoch = epochs.stop - 1
    global_loss, global_step = 0.0, 0.0

    # --- Training loop ---
    for epoch in epochs:
        sampler = getattr(train_loader, "sampler", None)
        if dist.is_initialized() and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        train_loss = _run_one_epoch(
            epoch, unet, train_loader, noise_scheduler, loss_pt, scale_factor,
            args.noise_scheduler["num_train_timesteps"], device, logger, local_rank, amp,
            args.diffusion_unet_train["freq_to_print"],
            args.diffusion_unet_train["conditional_free_guidance"],
            include_body_region, include_modality,
            optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler, is_train=True
        )

        val_loss = _run_one_epoch(
            epoch, unet, val_loader, noise_scheduler, loss_pt, scale_factor,
            args.noise_scheduler["num_train_timesteps"], device, logger, local_rank, amp,
            50, val_cfg_prob, include_body_region, include_modality, is_train=False
        )

        tl = train_loss[0].item() / train_loss[1].item()
        vl = val_loss[0].item() / val_loss[1].item() if val_loss[1].item() > 0 else float("inf")

        if (not dist.is_initialized()) or local_rank == 0:
            global_loss += train_loss[0].item()
            global_step += train_loss[1].item()
            logger.info(f"epoch {epoch+1} — train_loss: {tl:.4f}, val_loss: {vl:.4f}, global_loss: {global_loss/global_step:.4f}")

            should_save = ((epoch + 1) % save_every == 0) and (epoch != last_epoch)
            if should_save:
                save_checkpoint(epoch, unet, tl, args.noise_scheduler["num_train_timesteps"],
                                scale_factor, args.model_dir, args, val_loss=vl)

    # --- Final checkpoint with optimizer state ---
    if (not dist.is_initialized()) or local_rank == 0:
        save_checkpoint(epoch, unet, tl, args.noise_scheduler["num_train_timesteps"],
                        scale_factor, args.model_dir, args,
                        optimizer=optimizer, scheduler=lr_scheduler, grad_scaler=scaler, val_loss=vl)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument("--env_config", type=str, default="./configs/environment_diff_model_train.json")
    parser.add_argument("--model_config", type=str, default="./configs/config_diff_model.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_rflow.json")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--override_batch_size", type=int, default=None)
    parser.add_argument("--override_n_epochs", type=int, default=None)
    parser.add_argument("--override_n_epochs_total", type=int, default=None)
    parser.add_argument("--override_freq_to_print", type=int, default=None)
    parser.add_argument("--override_save_epoch_freq", type=int, default=None)
    parser.add_argument("--override_model_dir", type=str, default=None)
    parser.add_argument("--override_model_filename", type=str, default=None)
    parser.add_argument("--override_embedding_base_dir", type=str, default=None)
    parser.add_argument("--override_conditioning_base_dir", type=str, default=None)
    parser.add_argument("--override_log_dir", type=str, default=None)
    parser.add_argument("--override_existing_ckpt_filepath", type=str, default=None)
    parser.add_argument("--override_continue_training_from", type=int, default=None)
    cli_args = parser.parse_args()
    diff_model_train(
        cli_args.env_config,
        cli_args.model_config,
        cli_args.model_def,
        cli_args.num_gpus,
        cli_args.amp,
        runtime_args=cli_args,
    )
