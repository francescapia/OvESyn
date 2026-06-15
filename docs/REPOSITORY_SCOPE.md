# Repository Scope

This repository is a code release for the OvESyn method, not a data release.

## Public Components

- structured report generation pipeline
- CLIP adaptation scripts
- latent diffusion training and inference scripts
- evaluation utilities
- sanitized configuration templates
- synthetic schema examples

## Private Components

The following components remain outside the public repository:

- institutional CT cohort
- tumor and organ segmentations
- clinical metadata tables
- generated reports
- train/validation/test split files
- embeddings and latents
- trained institutional checkpoints
- per-case metrics and qualitative figures

The code assumes these assets exist locally under paths supplied by the user.
