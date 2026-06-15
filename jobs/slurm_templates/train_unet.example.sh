#!/usr/bin/env bash
#SBATCH --job-name=ovesyn_unet
#SBATCH --partition=<gpu_partition>
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/ovesyn_unet.%j.out
#SBATCH --error=logs/ovesyn_unet.%j.err

set -euo pipefail

module load CUDA/12.4.0
source /path/to/private/env/bin/activate

cd /path/to/OvESyn
export PYTHONPATH="$PWD:$PWD/scripts:${PYTHONPATH:-}"

torchrun --standalone --nnodes=1 --nproc_per_node=8 scripts/train_vanilla.py \
  --env_config /path/to/private_configs/unet_train_env.json \
  --model_config configs/templates/unet_model_train.example.json \
  --model_def configs/base/config_rflow.json \
  --num_gpus 8
