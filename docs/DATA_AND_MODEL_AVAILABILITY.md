# Data And Model Availability

## Suggested Manuscript Statement

The clinical CT cohort used in this study is not publicly available because it
contains private patient data governed by institutional approval, informed
consent, and applicable privacy regulations. Patient-level images,
segmentations, metadata, generated reports, train/validation/test split files,
and trained institutional checkpoints are therefore not released.

The source code for the OvESyn method is provided in a public repository to
support methodological inspection and reuse on appropriately governed local
datasets. The repository contains sanitized configuration templates and
synthetic schema examples, but no private data or patient-derived artifacts.

## Release Boundary

This public release may include:

- method source code
- prompt templates
- architecture configuration files
- synthetic examples of input schemas
- documentation for reproducing the workflow on private data

This public release must not include:

- CT volumes or segmentation masks
- patient IDs or linkable pseudonyms
- patient-level CSV, JSON, NPZ, or NIfTI files
- generated patient reports
- trained institutional model weights
- embeddings, latents, logs, run manifests, or generated figures

## Model Weights

Institutional fine-tuned weights are not included. Users should obtain any
third-party pretrained checkpoints from their official sources and verify that
their intended use complies with those licenses and terms.
