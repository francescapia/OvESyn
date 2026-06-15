# Public Release Checklist

Run this checklist before pushing any commit to a public remote.

## Data Disclosure

- [ ] No `data/`, `dataset/`, `results/`, `runs/`, `experiments/`, `models/`, or `checkpoints/` directories are committed.
- [ ] No `.nii`, `.nii.gz`, `.npz`, `.npy`, `.pt`, `.pth`, `.ckpt`, `.h5`, or generated `.csv` files are committed.
- [ ] No patient IDs, hospital-local IDs, or pseudonyms appear in code, docs, examples, logs, or comments.
- [ ] No generated reports, per-volume metrics, per-case radiomics, embeddings, latents, or manifests are committed.
- [ ] No paper figures derived from private cases are committed.

## Configuration

- [ ] Paths in committed configs are placeholders only.
- [ ] Cluster accounts, usernames, private mount points, and local workstation paths are absent.
- [ ] Any required pretrained checkpoints are referenced by source or placeholder, not included.

## Documentation

- [ ] README states that clinical data and institutional weights are not released.
- [ ] Data availability statement is consistent with the submitted manuscript.
- [ ] Example files contain synthetic values only.
- [ ] License choice has been approved by the authors/institution before public release.

## Suggested Local Audit Commands

```bash
rg -n "<replace-with-local-id-prefix>|<replace-with-private-mount>|<replace-with-local-username>" .
find . -type f \( -name "*.nii*" -o -name "*.npz" -o -name "*.npy" -o -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.h5" \)
git status --short
```

Review every match manually before pushing.
