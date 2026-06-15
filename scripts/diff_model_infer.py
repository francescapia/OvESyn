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
import logging
import os
import random
import json
from pathlib import Path
import sys
import copy

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


def load_filenames(data_list_path: str) -> list:
    """
    Load filenames from the JSON data list.

    Args:
        data_list_path (str): Path to the JSON data list file.

    Returns:
        list: List of filenames.
    """
    with open(data_list_path, "r") as file:
        json_data = json.load(file)
        for key in ("test", "validation", "training"):
            if key in json_data:
                filenames_raw = json_data[key]
                break
        else:
            raise KeyError(f"No recognized split key in {data_list_path}. Found: {list(json_data.keys())}")
    return [_item["image"] for _item in filenames_raw]


def set_random_seed(seed: int) -> int:
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.

    Returns:
        int: Set random seed.
    """
    random_seed = random.randint(0, 99999) if seed is None else seed
    set_determinism(random_seed)
    return random_seed


def _load_npy(path):
    return torch.tensor(np.load(path), dtype=torch.float32)


def load_models(args: argparse.Namespace, device: torch.device, logger: logging.Logger) -> tuple:
    """
    Load the autoencoder and UNet models.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load models on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: Loaded autoencoder, UNet model, and scale factor.
    """
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    try:
        checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
        autoencoder.load_state_dict(checkpoint_autoencoder)
    except Exception:
        logger.error("The trained_autoencoder_path does not exist!")

    unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint = torch.load(f"{args.model_dir}/{args.model_filename}", map_location=device, weights_only=False)
    if "unet_state_dict" in checkpoint.keys():
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=True)
        scale_factor = checkpoint["scale_factor"]

    else:
        unet.load_state_dict(checkpoint, strict=True)
        scale_factor = 1.0287

    logger.info(f"checkpoints {args.model_dir}/{args.model_filename} loaded.")
    logger.info(f"scale_factor -> {scale_factor}.")

    return autoencoder, unet, scale_factor


def prepare_tensors(args: argparse.Namespace, device: torch.device) -> tuple:
    """
    Prepare necessary tensors for inference.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load tensors on.

    Returns:
        tuple: Prepared top_region_index_tensor, bottom_region_index_tensor, and spacing_tensor.
    """
    top_region_index_tensor = np.array(args.diffusion_unet_inference["top_region_index"]).astype(float) * 1e2
    bottom_region_index_tensor = np.array(args.diffusion_unet_inference["bottom_region_index"]).astype(float) * 1e2
    spacing_tensor = np.array(args.diffusion_unet_inference["spacing"]).astype(float) * 1e2

    top_region_index_tensor = torch.from_numpy(top_region_index_tensor[np.newaxis, :]).half().to(device)
    bottom_region_index_tensor = torch.from_numpy(bottom_region_index_tensor[np.newaxis, :]).half().to(device)
    spacing_tensor = torch.from_numpy(spacing_tensor[np.newaxis, :]).half().to(device)
    modality_tensor = args.diffusion_unet_inference["modality"] * torch.ones((len(spacing_tensor)), dtype=torch.long).to(device)

    return top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor


def run_inference(
    args: argparse.Namespace,
    device: torch.device,
    autoencoder: torch.nn.Module,
    unet: torch.nn.Module,
    scale_factor: float,
    top_region_index_tensor: torch.Tensor,
    bottom_region_index_tensor: torch.Tensor,
    spacing_tensor: torch.Tensor,
    modality_tensor: torch.Tensor,
    output_size: tuple,
    divisor: int,
    logger: logging.Logger,
    impression: torch.Tensor = None
) -> np.ndarray:
    """
    Run the inference to generate synthetic images.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to run inference on.
        autoencoder (torch.nn.Module): Autoencoder model.
        unet (torch.nn.Module): UNet model.
        scale_factor (float): Scale factor for the model.
        top_region_index_tensor (torch.Tensor): Top region index tensor.
        bottom_region_index_tensor (torch.Tensor): Bottom region index tensor.
        spacing_tensor (torch.Tensor): Spacing tensor.
        modality_tensor (torch.Tensor): Modality tensor.
        output_size (tuple): Output size of the synthetic image.
        divisor (int): Divisor for downsample level.
        logger (logging.Logger): Logger for logging information.

    Returns:
        np.ndarray: Generated synthetic image data.
    """
    include_body_region = unet.include_top_region_index_input
    include_modality = unet.num_class_embeds is not None
    noise = torch.randn(
        (
            1,
            args.latent_channels,
            output_size[0] // divisor,
            output_size[1] // divisor,
            output_size[2] // divisor,
        ),
        device=device,
    )
    logger.info(f"noise: {noise.device}, {noise.dtype}, {type(noise)}")

    top_region_index_tensor = None if not args.diffusion_unet_def['include_top_region_index_input'] else top_region_index_tensor
    bottom_region_index_tensor = None if not args.diffusion_unet_def['include_top_region_index_input'] else bottom_region_index_tensor
    spacing_tensor = None if not args.diffusion_unet_def['include_top_region_index_input'] else spacing_tensor

    image = noise
    noise_scheduler = define_instance(args, "noise_scheduler")

    if isinstance(noise_scheduler, RFlowScheduler):
        noise_scheduler.set_timesteps(
            num_inference_steps=args.diffusion_unet_inference["num_inference_steps"],
            input_img_size_numel=torch.prod(torch.tensor(noise.shape[2:])),
        )
    else:
        noise_scheduler.set_timesteps(num_inference_steps=args.diffusion_unet_inference["num_inference_steps"])

    impression = impression.to(device)

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)
    autoencoder.eval()
    unet.eval()

    use_cfg = args.use_cfg
    guidance_scale = args.guidance_scale if use_cfg else 1.0  # Use CFG scale only when enabled

    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
    )

    with torch.amp.autocast("cuda", enabled=True):
        for t, next_t in progress_bar:

            unet_inputs = {
                "x": image,
                "timesteps": torch.Tensor((t,)).to(device),
                "spacing_tensor": spacing_tensor,
            }

            if include_body_region:
                unet_inputs.update(
                    {
                        "top_region_index_tensor": top_region_index_tensor,
                        "bottom_region_index_tensor": bottom_region_index_tensor,
                    }
                )

            if include_modality:
                unet_inputs.update(
                    {
                        "class_labels": modality_tensor,
                    }
                )

            if use_cfg:
                unet_inputs_no_text = copy.deepcopy(unet_inputs)
                unet_inputs.update(
                    {
                        "context": impression,
                    }
                )
                unet_inputs_no_text.update(
                    {
                        "context": torch.zeros_like(impression, device=device),
                    }
                )
                
                model_output_uncond = unet(
                    **unet_inputs_no_text
                )

                # Forward pass con condizione
                model_output_cond = unet(
                    **unet_inputs
                )

                # Formula CFG: combinazione tra output condizionato e non
                model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            
            else:
                unet_inputs.update(
                    {
                        "context": impression,
                    }
                )
                model_output = unet(
                    **unet_inputs
                )
            
            if not isinstance(noise_scheduler, RFlowScheduler):
                image, _ = noise_scheduler.step(model_output, t, image) 
            else:
                image, _ = noise_scheduler.step(model_output, t, image, next_t) 
    
        inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80],
            sw_batch_size=1,
            progress=True,
            mode="gaussian",
            overlap=0.4,
            sw_device=device,
            device=device,
        )
        synthetic_images = dynamic_infer(inferer, recon_model, image)

        data = synthetic_images.squeeze().cpu().detach().numpy()
        a_min, a_max, b_min, b_max = -1000, 1000, 0, 1
        data = (data - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
        data = np.clip(data, a_min, a_max)
        return np.int16(data)


def save_image(
    data: np.ndarray,
    output_size: tuple,
    out_spacing: tuple,
    output_path: str,
    logger: logging.Logger,
    resize: int = 512
) -> None:
    """
    Save the generated synthetic image to a file.

    Args:
        data (np.ndarray): Synthetic image data.
        output_size (tuple): Output size of the image.
        out_spacing (tuple): Spacing of the output image.
        output_path (str): Path to save the output image.
        logger (logging.Logger): Logger for logging information.
    """
    if resize != 512:
        from monai.transforms import Resized
        resize_transform = Resized(keys="image", spatial_size=(resize, resize), mode="trilinear")
        input = {'image': np.transpose(data, (2, 1, 0))}
        output = resize_transform(input)
        data = np.transpose(output['image'].numpy(), (2, 1, 0))

    out_affine = np.eye(4)
    for i in range(3):
        out_affine[i, i] = out_spacing[i]

    new_image = nib.Nifti1Image(data, affine=out_affine)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(new_image, output_path)
    logger.info(f"Saved {output_path}.")


@torch.inference_mode()
def diff_model_infer(env_config_path: str, model_config_path: str, model_def_path: str, num_gpus: int, index: int, resize: int) -> None:
    """
    Main function to run the diffusion model inference.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.
    """
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("inference")
    random_seed = set_random_seed(
        args.diffusion_unet_inference["random_seed"] + local_rank
        if args.diffusion_unet_inference["random_seed"]
        else None
    )
    logger.info(f"Using {device} of {world_size} with random seed: {random_seed}")

    filenames_raw = load_filenames(args.json_data_list)
    filenames_raw = filenames_raw[index:index+250]

    output_size = tuple(args.diffusion_unet_inference["dim"])
    out_spacing = tuple(args.diffusion_unet_inference["spacing"])

    autoencoder, unet, scale_factor = load_models(args, device, logger)
    num_downsample_level = max(
        1,
        (
            len(args.diffusion_unet_def["num_channels"])
            if isinstance(args.diffusion_unet_def["num_channels"], list)
            else len(args.diffusion_unet_def["attention_levels"])
        ),
    )
    divisor = 2 ** (num_downsample_level - 2)
    logger.info(f"num_downsample_level -> {num_downsample_level}, divisor -> {divisor}.")

    top_region_index_tensor, bottom_region_index_tensor, spacing_tensor, modality_tensor = prepare_tensors(args, device)
    check_input(None, None, None, output_size, out_spacing, None)

### CAMBIO PER POTER OPERARE SU DUE GPU
    #for p in filenames_raw: 
    for i, p in enumerate(filenames_raw):
        if (i % world_size) != local_rank:
            continue

        p_img = p if os.path.isabs(p) or p.startswith(args.data_base_dir) else os.path.join(args.data_base_dir, p)

        p_npy = (
            p_img.replace(args.data_base_dir, args.embedding_base_dir).replace(".nii.gz", f"_impression_{args.report_encoder_model}.npy")
        )
        cond = torch.Tensor(_load_npy(p_npy).unsqueeze(0))

        # path relativo dall'immagine
        rel = os.path.relpath(p_img, args.data_base_dir)  # es: 2016/nii/IEO.../ct.nii.gz
        parts = rel.split(os.sep)
        if len(parts) >= 3 and parts[1] == "nii":
            rel = os.path.join(parts[0], *parts[2:])     # es: 2016/IEO.../ct.nii.gz

        output_path = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        ckpt_filepath = f"{args.model_dir}/{args.model_filename}"

        if local_rank == 0:
            logger.info(f"[config] ckpt_filepath -> {ckpt_filepath}.")
            logger.info(f"[config] random_seed -> {random_seed}.")
            logger.info(f"[config] p_img -> {p_img}.")
            logger.info(f"[config] p_npy -> {p_npy}.")
            logger.info(f"[config] output_path -> {output_path}.")
            logger.info(f"[config] output_size -> {output_size}.")
            logger.info(f"[config] out_spacing -> {out_spacing}.")
            logger.info(f"[config] impression -> {cond.shape}.")

        data = run_inference(
            args,
            device,
            autoencoder,
            unet,
            scale_factor,
            top_region_index_tensor,
            bottom_region_index_tensor,
            spacing_tensor,
            modality_tensor,
            output_size,
            divisor,
            logger,
            cond,
        )

        save_image(data, output_size, out_spacing, output_path, logger, resize=resize)


    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Inference")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_diff_model_eval.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_diff_model.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_rflow.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for distributed inference",
    )
    parser.add_argument(
    "--index",
    type=int,
    default=0,
    )
    parser.add_argument(
    "--resize",
    type=int,
    default=512,
    )

    args = parser.parse_args()
    diff_model_infer(args.env_config, args.model_config, args.model_def, args.num_gpus, args.index, args.resize)
