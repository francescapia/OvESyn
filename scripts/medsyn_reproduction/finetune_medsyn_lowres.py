#!/usr/bin/env python3
"""Configurable low-resolution MedSyn fine-tuning wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_src", type=Path, required=True)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--prompt_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--train_num_steps", type=int, default=2000)
    ap.add_argument("--save_and_sample_every", type=int, default=500)
    ap.add_argument("--gradient_accumulate_every", type=int, default=4)
    ap.add_argument("--train_lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, str(args.source_src))
    import train_low_res as medsyn_low

    model = medsyn_low.Unet3D(
        dim=160,
        cond_dim=768,
        dim_mults=(1, 2, 4, 8),
        channels=4,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type="resnet",
        resnet_groups=8,
    )

    diffusion_model = medsyn_low.GaussianDiffusion(
        denoise_fn=model,
        image_size=64,
        num_frames=64,
        text_use_bert_cls=False,
        channels=4,
        timesteps=1000,
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.995,
        volume_depth=64,
        ddim_timesteps=50,
    )

    trainer = medsyn_low.Trainer(
        diffusion_model=diffusion_model,
        folder=args.data_dir,
        prompt_folder=args.prompt_dir,
        ema_decay=0.999,
        train_batch_size=args.batch_size,
        train_lr=args.train_lr,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        amp=True,
        step_start_ema=10000,
        update_ema_every=1,
        save_and_sample_every=args.save_and_sample_every,
        results_folder=args.save_dir,
        num_sample_rows=1,
        max_grad_norm=1.0,
    )

    if args.resume:
        trainer.load(-1)
    trainer.train()


if __name__ == "__main__":
    main()
