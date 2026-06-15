# Input Formats

This repository includes synthetic schema examples only. The values below are
fake and are not derived from clinical data.

## Metadata CSV

`prompt_generation/generate_captions_local_v4.py` expects one row per tumor
region. Multiple rows may share the same `patient_id` when a case has more than
one lesion label.

Required columns:

| Column | Meaning |
| --- | --- |
| `patient_id` | Local case identifier. Keep private. |
| `volume_name` | Case name used in split JSON and CLIP report CSV files. |
| `label` | Tumor label, typically `1` for omental and `9` for pelvic/ovarian. |
| `tumor_mask_path` | Path to the private tumor mask file. |
| `volume_ml` | Lesion volume in milliliters. |
| `mean_hu` | Mean lesion intensity in HU. |
| `p10_hu` | 10th percentile lesion intensity in HU. |
| `p90_hu` | 90th percentile lesion intensity in HU. |
| `organs_contact` | Comma-separated adjacent organs from segmentation. |
| `figo_stage` | FIGO stage used as routine clinical metadata. |
| `ascites` | `present` or `absent`. |

See [examples/sample_metadata.csv](../examples/sample_metadata.csv).

## Split JSON

The data list follows the MONAI-style split schema used by the training code:

```json
{
  "training": [
    {
      "image": "data/private_ct/CASE_SYN_0001/ct.nii.gz"
    }
  ],
  "validation": [],
  "test": []
}
```

Real split files should remain private because the case list and split
assignment can be linkable derived data.

## CLIP Report CSV

After report generation, `make_clip_reports_from_final.py` writes the report
schema used by CLIP training:

| Column | Meaning |
| --- | --- |
| `VolumeName` | Must match the case identifier used by the data loader. |
| `Findings_EN` | Generated structured Findings text. |
| `Impressions_EN` | Generated structured Impression text. |

Generated report CSVs are patient-derived and should not be committed.
