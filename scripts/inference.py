from __future__ import annotations

import argparse
import logging
import os
import random
import json
import re
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.utils import set_determinism

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from scripts.diff_model_setting import initialize_distributed, load_config, setup_logging
from scripts.sample import ReconModel, check_input
from scripts.utils import define_instance, dynamic_infer


LEGACY_DATA_PREFIX = "data/private_ct/"
PREPROCESSED_DATA_PREFIX = "data/private_ct_preprocessed/"
PROCESSED_CT_NAME = "ct_preprocessed.nii.gz"
RAW_CT_NAME = "ct.nii.gz"


def load_filenames(json_list_path: str):
    with open(json_list_path, "r") as f:
        json_data = json.load(f)
    for key in ["validation", "training", "test"]:
        if key in json_data:
            return json_data[key]
    raise KeyError("JSON must contain validation, training, or test keys.")


def _load_npy(path):
    return torch.tensor(np.load(path), dtype=torch.float32)


def _torch_load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _extract_unet_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["base_unet_state_dict", "unet_state_dict", "state_dict",
                   "model_state_dict", "model", "base_model", "base_unet", "pretrained_unet"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        prefixes = ("conv_in", "time_embed", "class_embedding", "spacing_layer",
                    "down_blocks", "middle_block", "up_blocks", "out.")
        sd = {k: v for k, v in ckpt.items() if isinstance(k, str) and k.startswith(prefixes)}
        if sd:
            return sd
        return {}
    return ckpt


def _extract_path(item):
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in ["image", "img", "ct", "path", "filepath", "real_path"]:
            if k in item:
                return item[k]
    raise TypeError("Unsupported filename entry type")


def _normalize_rel_path(path_str: str) -> str:
    value = Path(path_str).as_posix()
    if value.startswith("./"):
        value = value[2:]
    return value


def _extract_relative_volume_path(path_str: str) -> str:
    posix = _normalize_rel_path(path_str)

    for marker in (LEGACY_DATA_PREFIX, PREPROCESSED_DATA_PREFIX):
        if marker in posix:
            return posix.split(marker, 1)[1]

    parts = Path(posix).parts
    if len(parts) >= 4 and parts[-3] == "nii":
        return Path(*parts[-4:]).as_posix()

    raise ValueError(f"Cannot derive relative volume path from: {path_str}")


def _resolve_input_image_path(path_str: str, args) -> str:
    p = Path(path_str)
    if p.is_absolute():
        candidates = [p]
        if p.name == PROCESSED_CT_NAME:
            candidates.append(p.with_name(RAW_CT_NAME))
        for candidate in candidates:
            if candidate.exists():
                return candidate.as_posix()
        return p.as_posix()

    norm = _normalize_rel_path(path_str)
    candidates: list[Path] = []
    if norm.startswith(LEGACY_DATA_PREFIX):
        rel = norm[len(LEGACY_DATA_PREFIX):].lstrip("/")
        candidates.append(ROOT / norm)
        if rel.endswith(PROCESSED_CT_NAME):
            candidates.append(ROOT / LEGACY_DATA_PREFIX / rel.replace(PROCESSED_CT_NAME, RAW_CT_NAME))
        candidates.append(ROOT / PREPROCESSED_DATA_PREFIX / rel)
    elif norm.startswith(PREPROCESSED_DATA_PREFIX):
        candidates.append(ROOT / norm)
        if norm.endswith(PROCESSED_CT_NAME):
            rel = norm[len(PREPROCESSED_DATA_PREFIX):].lstrip("/")
            candidates.append(ROOT / LEGACY_DATA_PREFIX / rel.replace(PROCESSED_CT_NAME, RAW_CT_NAME))
    else:
        base = _normalize_rel_path(args.data_base_dir)
        if base and norm.startswith(base + "/"):
            candidates.append(ROOT / norm)
        else:
            candidates.append(ROOT / base / norm)

    for candidate in candidates:
        if candidate.exists():
            return candidate.as_posix()
    return candidates[0].as_posix()


def _checkpoint_prefix(model_filename: str) -> str:
    stem = Path(model_filename).stem
    m = re.match(r"^(.*?)(?:_\d+(?:_resume)?)?$", stem)
    return m.group(1) if m else stem


def resolve_unet_checkpoint(args, checkpoint_epoch: int | None) -> tuple[str, str]:
    model_dir = Path(args.model_dir)

    if checkpoint_epoch is None:
        ckpt_path = (model_dir / args.model_filename)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"UNet checkpoint not found: {ckpt_path}")
        return str(ckpt_path), ckpt_path.stem

    prefix = _checkpoint_prefix(args.model_filename)
    candidates = [
        model_dir / f"{prefix}_{checkpoint_epoch}.pt",
        model_dir / f"{prefix}_{checkpoint_epoch}_resume.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate), candidate.stem

    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Cannot find checkpoint for epoch {checkpoint_epoch}. Tried:\n{tried}"
    )


def apply_runtime_overrides(args, runtime_args) -> None:
    if runtime_args is None:
        return
    if getattr(runtime_args, "override_model_dir", None) is not None:
        args.model_dir = runtime_args.override_model_dir
    if getattr(runtime_args, "override_model_filename", None) is not None:
        args.model_filename = runtime_args.override_model_filename
    if getattr(runtime_args, "override_embedding_base_dir", None) is not None:
        args.embedding_base_dir = runtime_args.override_embedding_base_dir
    if getattr(runtime_args, "override_conditioning_base_dir", None) is not None:
        args.conditioning_base_dir = runtime_args.override_conditioning_base_dir
    if getattr(runtime_args, "override_output_dir", None) is not None:
        args.output_dir = runtime_args.override_output_dir


def load_models(args, device, logger):
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    ckpt_ae = _torch_load(args.trained_autoencoder_path)
    if isinstance(ckpt_ae, dict) and "state_dict" in ckpt_ae:
        ckpt_ae = ckpt_ae["state_dict"]
    autoencoder.load_state_dict(ckpt_ae, strict=True)

    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet_ckpt_path = getattr(args, "resolved_unet_ckpt_path", os.path.join(args.model_dir, args.model_filename))
    ckpt = _torch_load(unet_ckpt_path)

    ########## DEBUG #########
    logger.info(f"UNet ckpt: {unet_ckpt_path}")
    if isinstance(ckpt, dict):
        logger.info(f"ckpt keys: {list(ckpt.keys())[:20]}")
        for k in ["epoch", "global_step", "best_metric", "val_loss"]:
            if k in ckpt:
                logger.info(f"{k}: {ckpt[k]}")
    ###########################
    scale_factor = ckpt.get("scale_factor", 1.0287) if isinstance(ckpt, dict) else 1.0287

    if isinstance(ckpt, dict) and "lora_state_dict" in ckpt:
        base_path = getattr(args, "existing_ckpt_filepath", None)
        if base_path is None:
            raise ValueError("Missing args.existing_ckpt_filepath for LoRA")

        base_ckpt = ckpt if base_path == unet_ckpt_path else _torch_load(base_path)
        base_sd = _extract_unet_state_dict(base_ckpt)
        if len(base_sd) == 0:
            raise RuntimeError("FATAL: Base checkpoint missing or invalid.")

        unet.load_state_dict(base_sd, strict=True)

        from peft import LoraConfig, get_peft_model
        lora_cfg = ckpt.get("lora_config", None)
        if isinstance(lora_cfg, dict):
            mapped = {
                "r": lora_cfg.get("rank", 8), "lora_alpha": lora_cfg.get("alpha", 16),
                "lora_dropout": lora_cfg.get("dropout", 0.0),
                "target_modules": lora_cfg.get("target_modules", ["to_q", "to_k", "to_v"]),
                "bias": "none", "task_type": "FEATURE_EXTRACTION",
            }
            lora_cfg = LoraConfig(**mapped)
        else:
            lora_cfg = LoraConfig(
                r=getattr(args, "lora_rank", 8), lora_alpha=getattr(args, "lora_alpha", 16),
                target_modules=getattr(args, "lora_target_modules", ["to_q", "to_k", "to_v"]),
                bias="none", task_type="FEATURE_EXTRACTION",
            )

        unet = get_peft_model(unet, lora_cfg)
        unet.load_state_dict(ckpt["lora_state_dict"], strict=False)
        unet = unet.merge_and_unload().to(device)
        logger.info("LoRA loaded and merged.")
    else:
        unet.load_state_dict(_extract_unet_state_dict(ckpt), strict=True)

    autoencoder.eval()
    unet.eval()
    return autoencoder, unet, scale_factor


def prepare_tensors(args, device):
    inf = args.diffusion_unet_inference
    top = torch.from_numpy(np.array(inf["top_region_index"]) * 1e2).unsqueeze(0).half().to(device)
    bot = torch.from_numpy(np.array(inf["bottom_region_index"]) * 1e2).unsqueeze(0).half().to(device)
    spc = torch.from_numpy(np.array(inf["spacing"]) * 1e2).unsqueeze(0).half().to(device)
    mod = inf["modality"] * torch.ones(len(spc), dtype=torch.long, device=device)
    return top, bot, spc, mod


def load_batch_impressions(paths, args):
    emb_base = Path(getattr(args, "conditioning_base_dir", args.embedding_base_dir)).as_posix().lstrip("./")

    embs = []
    for p in paths:
        rel = _extract_relative_volume_path(p)
        npy = f"{emb_base}/{rel}"
        npy = npy.replace(".nii.gz", f"_impression_{args.report_encoder_model}.npy")
        embs.append(_load_npy(npy))
    return torch.stack(embs)


def run_inference_batch(
    args, device, autoencoder, unet, scale_factor,
    top_region, bottom_region, spacing, modality,
    output_size, divisor, impressions
):
    bs = impressions.shape[0]
    include_body = unet.include_top_region_index_input
    include_mod = unet.num_class_embeds is not None

    lat_shape = (bs, args.latent_channels,
                 output_size[0] // divisor, output_size[1] // divisor, output_size[2] // divisor)
    
    ######### DEBUG ########
    print("output_size:", output_size, "divisor:", divisor, "latent_channels:", args.latent_channels, "lat_shape:", lat_shape)
    ########################
    
    image = torch.randn(lat_shape, device=device)

    top_r = top_region.repeat(bs, 1) if include_body else None
    bot_r = bottom_region.repeat(bs, 1) if include_body else None
    spc = spacing.repeat(bs, 1)
    mod = modality.repeat(bs) if include_mod else None

    noise_scheduler = define_instance(args, "noise_scheduler")
    inf = args.diffusion_unet_inference

    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=inf["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(image.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=inf["num_inference_steps"])

    impressions = impressions.to(device)
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    use_cfg = getattr(args, "use_cfg", False)
    guidance_scale = getattr(args, "guidance_scale", 1.0) if use_cfg else 1.0

    all_t = noise_scheduler.timesteps
    all_next_t = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype)))

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in tqdm(zip(all_t, all_next_t), total=len(all_t), leave=False):
            t_batch = torch.full((bs,), t.item(), device=device)
            inputs = {"x": image, "timesteps": t_batch, "spacing_tensor": spc}

            if include_body:
                inputs["top_region_index_tensor"] = top_r
                inputs["bottom_region_index_tensor"] = bot_r
            if include_mod:
                inputs["class_labels"] = mod

            if use_cfg:
                inp_cond, inp_uncond = dict(inputs), dict(inputs)
                inp_cond["context"] = impressions
                inp_uncond["context"] = torch.zeros_like(impressions)
                out_u = unet(**inp_uncond)
                out_c = unet(**inp_cond)
                model_output = out_u + guidance_scale * (out_c - out_u)
            else:
                inputs["context"] = impressions
                model_output = unet(**inputs)

            if isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image, next_t)
            else:
                image, _ = noise_scheduler.step(model_output, t, image)

        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80], sw_batch_size=1, mode="gaussian",
            overlap=0.4, sw_device=device, device=device,
        )

        results = []
        for i in range(bs):
            syn = dynamic_infer(inferer, recon_model, image[i:i + 1]).squeeze().cpu().detach().numpy()
            # results = []
            # for i in range(bs):
            #     syn = dynamic_infer(inferer, recon_model, image[i:i + 1]).squeeze().cpu().detach().numpy()

            #     # ===== DEBUG OUTPUT =====
            #     logger.info(f"\n--- Volume {i} ---")
            #     logger.info(f"Raw output - min: {syn.min():.6f}, max: {syn.max():.6f}, mean: {syn.mean():.6f}, std: {syn.std():.6f}")
            #     logger.info(f"Data type: {syn.dtype}")

            #     # Conta valori fuori range
            #     out_of_range = np.sum((syn < 0) | (syn > 1))
            #     logger.info(f"Values outside [0,1]: {out_of_range} / {syn.size} ({100*out_of_range/syn.size:.2f}%)")

            #     # Istogramma
            #     hist, bins = np.histogram(syn, bins=10, range=(0, 1))
            #     logger.info(f"Histogram: {hist}")

            #     # Clipping
            #     syn_clipped = np.clip(syn, 0.0, 1.0).astype(np.float32)
            #     logger.info(f"After clipping - min: {syn_clipped.min():.6f}, max: {syn_clipped.max():.6f}")

            #     results.append(syn_clipped)
            # CAMBIO PER COERENZA CON DATI REAL 
            #syn = np.clip(syn * 2000 - 1000, -1000, 1000)
            #results.append(np.int16(syn))
            syn = np.clip(syn, 0.0, 1.0).astype(np.float32)
            results.append(syn)

    return results


def save_image(data, output_size, out_spacing, output_path, resize=512):
    if resize != 512:
        from monai.transforms import Resized
        data = np.transpose(
            Resized(keys="image", spatial_size=(resize, resize), mode="trilinear")(
                {"image": np.transpose(data, (2, 1, 0))}
            )["image"].numpy(), (2, 1, 0),
        )

    affine = np.eye(4)
    for i in range(3):
        affine[i, i] = out_spacing[i]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=affine), output_path)


@torch.inference_mode()
def diff_model_infer(
    env_config,
    model_config,
    model_def,
    num_gpus,
    index,
    resize,
    batch_size,
    checkpoint_epoch=None,
    runtime_args=None,
):
    args = load_config(env_config, model_config, model_def)
    apply_runtime_overrides(args, runtime_args)
    if not getattr(args, "conditioning_base_dir", None):
        args.conditioning_base_dir = args.embedding_base_dir
    args.checkpoint_epoch = checkpoint_epoch
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")

    seed = args.diffusion_unet_inference.get("random_seed")
    if seed is not None:
        set_determinism(seed + local_rank)
    else:
        set_determinism(random.randint(0, 99999))

    resolved_ckpt_path, ckpt_tag = resolve_unet_checkpoint(args, getattr(args, "checkpoint_epoch", None))
    args.resolved_unet_ckpt_path = resolved_ckpt_path
    args.output_dir = str(Path(args.output_dir) / ckpt_tag)
    logger.info(f"Resolved checkpoint: {resolved_ckpt_path}")
    logger.info(f"Output directory for this run: {args.output_dir}")

    filenames_raw = load_filenames(args.json_data_list)[index:]
    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    n_levels = (len(args.diffusion_unet_def["num_channels"])
                if isinstance(args.diffusion_unet_def["num_channels"], list)
                else len(args.diffusion_unet_def["attention_levels"]))
    divisor = 2 ** (max(1, n_levels) - 2)
    top_t, bot_t, spc_t, mod_t = prepare_tensors(args, device)
    check_input(None, None, None, output_size, out_spacing, None)

    files_gpu = [p for i, p in enumerate(filenames_raw) if (i % world_size) == local_rank]

    for batch_idx in tqdm(range((len(files_gpu) + batch_size - 1) // batch_size), position=local_rank):
        batch_files = files_gpu[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        img_paths, out_paths = [], []

        for p in batch_files:
            p_str = _extract_path(p)
            p_img = _resolve_input_image_path(p_str, args)
            rel = _extract_relative_volume_path(p_img)
            out = os.path.join(args.output_dir, rel)

            if not os.path.exists(out):
                img_paths.append(p_img)
                out_paths.append(out)

        if not img_paths:
            continue

        impressions = load_batch_impressions(img_paths, args)
        results = run_inference_batch(
            args, device, autoencoder, unet, scale_factor,
            top_t, bot_t, spc_t, mod_t, output_size, divisor, impressions,
        )

        for data, path in zip(results, out_paths):
            save_image(data, output_size, out_spacing, path, resize)

        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_config", type=str, default="./configs/environment_diff_model_eval.json")
    parser.add_argument("--model_config", type=str, default="./configs/config_diff_model.json")
    parser.add_argument("--model_def", type=str, default="./configs/config_rflow.json")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--checkpoint_epoch",
        type=int,
        default=None,
        help="Optional epoch number to select checkpoint file automatically (e.g., 50 -> *_50.pt)",
    )
    parser.add_argument("--override_model_dir", type=str, default=None)
    parser.add_argument("--override_model_filename", type=str, default=None)
    parser.add_argument("--override_embedding_base_dir", type=str, default=None)
    parser.add_argument("--override_conditioning_base_dir", type=str, default=None)
    parser.add_argument("--override_output_dir", type=str, default=None)
    args = parser.parse_args()
    diff_model_infer(args.env_config, args.model_config, args.model_def,
                     args.num_gpus, args.index, args.resize, args.batch_size,
                     checkpoint_epoch=args.checkpoint_epoch, runtime_args=args)
