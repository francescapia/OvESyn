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

import torch
import torch.distributed as dist

from monai.utils import RankFilter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def setup_logging(logger_name: str = "") -> logging.Logger:
    """
    Setup the logging configuration.

    Args:
        logger_name (str): logger name.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(logger_name)
    if dist.is_initialized():
        logger.addFilter(RankFilter())
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logger


def load_config(env_config_path: str, model_config_path: str, model_def_path: str) -> argparse.Namespace:
    """
    Load configuration from JSON files.

    Args:
        env_config_path (str): Path to the environment configuration file.
        model_config_path (str): Path to the model configuration file.
        model_def_path (str): Path to the model definition file.

    Returns:
        argparse.Namespace: Loaded configuration.
    """
    args = argparse.Namespace()

    with open(env_config_path, "r") as f:
        env_config = json.load(f)
    for k, v in env_config.items():
        setattr(args, k, v)

    with open(model_config_path, "r") as f:
        model_config = json.load(f)
    for k, v in model_config.items():
        setattr(args, k, v)

    with open(model_def_path, "r") as f:
        model_def = json.load(f)
    for k, v in model_def.items():
        setattr(args, k, v)

    return args
    
def initialize_distributed(num_gpus: int) -> tuple:
    """
    Initialize distributed training.

    Returns:
        tuple: local_rank, world_size, and device.
    """
    if torch.cuda.is_available():
        local_rank_env = os.environ.get("LOCAL_RANK")
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        launched_with_torchrun = local_rank_env is not None or world_size_env > 1

        if launched_with_torchrun:
            local_rank = int(local_rank_env or 0)
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            if not dist.is_initialized():
                try:
                    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
                except TypeError:
                    dist.init_process_group(backend="nccl", init_method="env://")
            world_size = dist.get_world_size()
            return local_rank, world_size, device

        if num_gpus > 1:
            raise RuntimeError(
                "Multi-GPU training requires a distributed launcher such as "
                "`torchrun --standalone --nproc_per_node=<N>`."
            )

        local_rank = 0
        world_size = 1
        device = torch.device("cuda")
    else:
        local_rank = 0
        world_size = 1
        device = torch.device("cpu")
    return local_rank, world_size, device


def distributed_barrier(device: torch.device | int | None = None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return

    if torch.cuda.is_available():
        if isinstance(device, torch.device):
            device_id = device.index
        elif isinstance(device, int):
            device_id = device
        else:
            device_id = torch.cuda.current_device()
        if device_id is not None:
            try:
                dist.barrier(device_ids=[device_id])
                return
            except TypeError:
                pass

    dist.barrier()
