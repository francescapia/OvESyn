#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - only used in minimal environments
    def tqdm(iterable, **_kwargs):
        return iterable


LABEL_TO_LOCATION = {
    1: "omental",
    9: "pelvic/ovarian",
}

IMPRESSION_LOCATION = {
    "omental": "omental",
    "pelvic/ovarian": "adnexal/pelvic",
}

FIELDNAMES = [
    "prompt_version",
    "patient_id",
    "volume_name",
    "labels",
    "findings",
    "impressions",
    "combined_report",
    "figo_stage",
    "ascites",
    "total_volume_ml",
    "n_tumors",
]


def clean_text(value: object) -> str:
    return " ".join(str(value).split()).strip()


def normalize_ascites(value: object) -> str:
    text = clean_text(value).lower()
    if text in {"present", "yes", "y", "true", "1"}:
        return "present"
    if text in {"absent", "no", "n", "false", "0"}:
        return "absent"
    return "absent"


def normalize_figo(value: object) -> str:
    text = clean_text(value).upper()
    return text or "III"


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_path(path_str: str, repo_root: Path, metadata_dir: Path) -> Path:
    path = Path(clean_text(path_str))
    if path.is_absolute():
        return path
    for candidate in (repo_root / path, metadata_dir / path):
        if candidate.exists():
            return candidate.resolve()
    return (repo_root / path).resolve()


def load_ids_for_key(json_path: Path, key: str) -> set[str]:
    data = json.loads(json_path.read_text())
    return {Path(item["image"]).parent.name for item in data.get(key, [])}


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return set(zip(*(tokens[i:] for i in range(n))))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:/[a-z0-9]+)?", text.lower())


def quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0}

    metrics = []
    for row in rows:
        findings = str(row.get("findings", ""))
        impression = str(row.get("impressions", ""))
        f_tokens = tokenize(findings)
        i_tokens = tokenize(impression)
        f_set = set(f_tokens)
        i_set = set(i_tokens)
        f_bigrams = ngrams(f_tokens, 2)
        i_bigrams = ngrams(i_tokens, 2)
        f_trigrams = ngrams(f_tokens, 3)
        i_trigrams = ngrams(i_tokens, 3)
        metrics.append(
            {
                "impression_len": len(i_tokens),
                "findings_len": len(f_tokens),
                "impr_unigram_in_findings": sum(1 for t in i_tokens if t in f_set) / max(1, len(i_tokens)),
                "impr_bigram_in_findings": len(i_bigrams & f_bigrams) / max(1, len(i_bigrams)),
                "impr_trigram_in_findings": len(i_trigrams & f_trigrams) / max(1, len(i_trigrams)),
                "unigram_jaccard": len(f_set & i_set) / max(1, len(f_set | i_set)),
            }
        )

    summary: dict[str, Any] = {"n_rows": len(rows)}
    for key in metrics[0]:
        values = [m[key] for m in metrics]
        summary[key] = {
            "mean": round(mean(values), 4),
            "median": round(median(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }
    summary["high_overlap_counts"] = {
        "unigram_in_findings_ge_0_70": sum(m["impr_unigram_in_findings"] >= 0.70 for m in metrics),
        "unigram_in_findings_ge_0_80": sum(m["impr_unigram_in_findings"] >= 0.80 for m in metrics),
        "trigram_in_findings_ge_0_40": sum(m["impr_trigram_in_findings"] >= 0.40 for m in metrics),
        "trigram_in_findings_ge_0_50": sum(m["impr_trigram_in_findings"] >= 0.50 for m in metrics),
    }
    return summary


class StructuredImpressionReportGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        from vllm import LLM, SamplingParams

        print(f"Loading {model_name}...")
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True,
        )

        self.findings_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=400,
            repetition_penalty=1.1,
            stop=["</FINDINGS>", "\n\n\n"],
        )

        self.impressions_params = SamplingParams(
            temperature=0.25,
            top_p=0.9,
            max_tokens=120,
            repetition_penalty=1.18,
            stop=["</IMPRESSION>", "\n\n\n"],
        )

    def extract_shape_radiomics(self, mask_path: Path, label: int) -> dict[str, float]:
        import SimpleITK as sitk
        from radiomics import featureextractor

        mask_img = sitk.ReadImage(str(mask_path))
        mask_bin = sitk.BinaryThreshold(
            mask_img,
            lowerThreshold=int(label),
            upperThreshold=int(label),
            insideValue=1,
            outsideValue=0,
        )

        arr = sitk.GetArrayViewFromImage(mask_bin)
        if arr.sum() == 0:
            raise ValueError(f"Empty mask for label={label} at {mask_path}")

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.disableAllImageTypes()
        extractor.enableFeatureClassByName("shape")

        features = extractor.execute(mask_bin, mask_bin)

        return {
            "sphericity": float(features["original_shape_Sphericity"]),
            "elongation": float(features["original_shape_Elongation"]),
            "flatness": float(features["original_shape_Flatness"]),
            "max_diameter_mm": float(features["original_shape_Maximum3DDiameter"]),
            "surface_area_mm2": float(features["original_shape_SurfaceArea"]),
            "volume_mm3": float(features["original_shape_VoxelVolume"]),
        }

    def categorize_features(self, patient_data: dict[str, Any], radiomics: dict[str, float]) -> dict[str, Any]:
        max_diameter_cm = radiomics["max_diameter_mm"] / 10
        if max_diameter_cm < 3:
            size_desc = "small"
            size_cm = f"{max_diameter_cm:.1f} cm"
        elif max_diameter_cm < 5:
            size_desc = "small-to-moderate"
            size_cm = f"{max_diameter_cm:.1f} cm"
        elif max_diameter_cm < 10:
            size_desc = "moderate-sized"
            size_cm = f"{max_diameter_cm:.1f} cm"
        elif max_diameter_cm < 15:
            size_desc = "large"
            size_cm = f"{max_diameter_cm:.1f} cm"
        else:
            size_desc = "very large"
            size_cm = f"{max_diameter_cm:.0f} cm"

        sph = radiomics["sphericity"]
        elong = radiomics["elongation"]
        flat = radiomics["flatness"]

        if sph > 0.90:
            shape_desc = "spherical"
        elif sph > 0.70:
            shape_desc = "ovoid"
        elif sph > 0.50:
            shape_desc = "lobulated"
        else:
            shape_desc = "irregular, multilobulated"

        if elong > 0.70:
            shape_desc += ", elongated"
        if flat > 0.60:
            shape_desc += ", flattened"

        mean_hu = patient_data["mean_hu"]
        p10_hu = patient_data["p10_hu"]
        p90_hu = patient_data["p90_hu"]
        hu_range = p90_hu - p10_hu

        if mean_hu < 20:
            density_primary = "simple cystic with fluid density"
        elif mean_hu < 40:
            density_primary = "complex cystic"
        elif mean_hu < 60:
            density_primary = "mixed solid and cystic"
        else:
            density_primary = "predominantly solid"

        if hu_range > 70:
            heterogeneity = "markedly heterogeneous"
        elif hu_range > 50:
            heterogeneity = "moderately heterogeneous"
        elif hu_range > 30:
            heterogeneity = "mildly heterogeneous"
        else:
            heterogeneity = "relatively homogeneous"

        return {
            "size_desc": size_desc,
            "size_cm": size_cm,
            "shape_desc": shape_desc,
            "density_primary": density_primary,
            "heterogeneity": heterogeneity,
            "has_low_density": p10_hu < 10 and hu_range > 40,
            "sphericity": sph,
            "mean_hu": mean_hu,
        }

    def build_findings_prompt(self, tumors_data: list[dict[str, Any]], figo_stage: str, ascites: str, total_volume_ml: float) -> str:
        example = """<FINDINGS>
An irregular multilobulated omental mass measuring 14.9 cm with volume 31.2 mL demonstrates predominantly solid attenuation with moderate heterogeneity. The mass is adjacent to small bowel and colon. A separate lobulated pelvic/ovarian mass measuring 7.6 cm with volume 7.6 mL shows mixed solid and cystic components with mild heterogeneity, adjacent to bladder. Combined tumor burden is 38.8 mL. Ascites is present. Findings are consistent with FIGO Stage III disease.
</FINDINGS>"""

        tumor_descriptions = []
        for i, tumor in enumerate(tumors_data, 1):
            categorized = tumor["categorized"]
            patient_data = tumor["patient_data"]
            desc = f"""Tumor {i} ({tumor['location']}):
- Size: {categorized['size_desc']} measuring {categorized['size_cm']}
- Individual volume: {tumor['volume_ml']:.1f} mL
- Morphology: {categorized['shape_desc']}
- Attenuation: {categorized['density_primary']}, {categorized['heterogeneity']}
- Mean HU: {categorized['mean_hu']:.0f} (range {patient_data['p10_hu']:.0f}-{patient_data['p90_hu']:.0f})"""

            if categorized["has_low_density"]:
                desc += "\n- Low attenuation areas present"

            organs = ", ".join(patient_data["organs_contact"][:4])
            if organs:
                desc += f"\n- Adjacent organs: {organs}"
            tumor_descriptions.append(desc)

        tumors_text = "\n\n".join(tumor_descriptions)
        ascites_instruction = "State: 'Ascites is present.'" if ascites == "present" else "State: 'No ascites is identified.'"

        return f"""You are a radiologist writing objective CT findings. Output ONLY the findings text within <FINDINGS></FINDINGS> tags.

CRITICAL RULES - NO EXCEPTIONS:
1. NEVER use: "invades", "invasion", "metastatic", "malignancy", "peritoneal implants"
2. For organ proximity, use ONLY: "is adjacent to [organ]" or "adjacent to [organ]" (do not imply invasion)
3. Ascites: {ascites_instruction} (NEVER "mild/moderate/severe")
4. FIGO: State only "Findings are consistent with FIGO Stage {figo_stage} disease." (do not elaborate)
5. Use ONLY information from PATIENT DATA below - no additional clinical interpretation
6. Report individual volumes AND combined total

EXAMPLE:
{example}

PATIENT DATA:
{tumors_text}

Combined tumor burden: {total_volume_ml:.1f} mL
FIGO stage: {figo_stage}
Ascites: {ascites}

<FINDINGS>"""

    def build_impression_key_data(self, tumors_data: list[dict[str, Any]], figo_stage: str, ascites: str, total_volume_ml: float) -> dict[str, str]:
        sites = [IMPRESSION_LOCATION.get(t["location"], t["location"]) for t in tumors_data]
        unique_sites = []
        for site in sites:
            if site not in unique_sites:
                unique_sites.append(site)
        sites_text = " and ".join(unique_sites) if unique_sites else "not provided"

        dominant = max(tumors_data, key=lambda t: t["volume_ml"])
        dominant_region = IMPRESSION_LOCATION.get(dominant["location"], dominant["location"])
        if len(tumors_data) == 1:
            overall_distribution = "localized"
        elif len(unique_sites) >= 2 and total_volume_ml >= 250:
            overall_distribution = "extensive multifocal"
        elif len(unique_sites) >= 2:
            overall_distribution = "multifocal"
        elif total_volume_ml >= 250:
            overall_distribution = "extensive localized"
        else:
            overall_distribution = "limited"

        density_by_volume = sorted(
            tumors_data,
            key=lambda t: t["volume_ml"],
            reverse=True,
        )
        densities = []
        for tumor in density_by_volume:
            density = self.impression_density(tumor["categorized"]["density_primary"])
            if density not in densities:
                densities.append(density)
        dominant_attenuation = " and ".join(densities[:2]) if densities else "not provided"

        heterogeneity = self.impression_heterogeneity([t["categorized"]["heterogeneity"] for t in tumors_data])

        key_data = {
            "sites": sites_text,
            "number_of_tumor_regions": str(len(tumors_data)),
            "total_tumor_burden": f"{total_volume_ml:.1f}",
            "dominant_region": dominant_region,
            "overall_distribution": overall_distribution,
            "dominant_attenuation_pattern": dominant_attenuation,
            "heterogeneity": heterogeneity,
            "ascites_status": ascites,
            "figo_stage": figo_stage,
        }
        key_data["interpretive_pattern"] = self.impression_interpretive_pattern(key_data)
        return key_data

    @staticmethod
    def impression_interpretive_pattern(key_data: dict[str, str]) -> str:
        sites = key_data["sites"]
        figo = key_data["figo_stage"]
        ascites = key_data["ascites_status"]
        distribution = key_data["overall_distribution"]

        if "omental" in sites and "adnexal/pelvic" in sites and figo in {"III", "IIIC"}:
            if ascites == "present":
                return "advanced omental and adnexal/pelvic disease pattern with ascites"
            return "advanced omental and adnexal/pelvic disease pattern"

        if "adnexal/pelvic" in sites and "omental" not in sites and figo in {"I", "II"}:
            return "localized adnexal/pelvic disease pattern"

        if distribution == "extensive multifocal":
            return "extensive multifocal tumor burden pattern"
        if distribution == "extensive localized":
            return "extensive localized tumor burden pattern"

        return "tumor burden pattern"

    @staticmethod
    def impression_density(value: str) -> str:
        mapping = {
            "simple cystic with fluid density": "simple cystic",
            "complex cystic": "complex cystic",
            "mixed solid and cystic": "mixed solid-cystic",
            "predominantly solid": "predominantly solid",
        }
        return mapping.get(value, value)

    @staticmethod
    def impression_heterogeneity(values: list[str]) -> str:
        rank = {
            "relatively homogeneous": (0, "low"),
            "mildly heterogeneous": (1, "mild"),
            "moderately heterogeneous": (2, "moderate"),
            "markedly heterogeneous": (3, "marked"),
        }
        best = max((rank.get(v, (0, clean_text(v)))[0] for v in values), default=0)
        for _label, (score, text) in rank.items():
            if score == best:
                return text
        return "not provided"

    def build_impressions_prompt(self, tumors_data: list[dict[str, Any]], figo_stage: str, ascites: str, total_volume_ml: float) -> str:
        key_data = self.build_impression_key_data(tumors_data, figo_stage, ascites, total_volume_ml)

        return f"""You are writing the IMPRESSION section of a CT radiology report from structured imaging descriptors for ovarian cancer assessment.

Output ONLY within <IMPRESSION></IMPRESSION> tags.

TASK:
Generate a concise radiology-style IMPRESSION based only on the KEY DATA provided.

RULES:
1. Summarize the clinically relevant interpretation; do not restate the findings line-by-line.
2. Write 1-2 sentences, 25-45 words total.
3. Mention overall distribution, total tumor burden, ascites status, and FIGO stage.
4. Do not mention individual lesion sizes, individual lesion volumes, HU values, organ-by-organ adjacency, or measurement-by-measurement details.
5. Use "adnexal/pelvic" instead of "pelvis/ovaries".
6. Do not add information not present in KEY DATA.
7. Do not diagnose histology, tissue invasion, metastases, or metastatic disease unless explicitly included in KEY DATA.
8. Avoid artificial phrases such as "imaging descriptors show", "clinical correlation is", "the key data indicate", and "findings are consistent with".
9. If ascites is absent, state "without ascites" or "no ascites"; if present, state "with ascites".
10. If attenuation or heterogeneity descriptors differ across regions, summarize the overall pattern rather than listing every component.

STYLE:
Use natural CT radiology language: concise, interpretive, and non-repetitive.

EXAMPLE 1:
KEY DATA:
- Sites: omental and adnexal/pelvic
- Number of tumor regions: 2
- Total tumor burden: 38.8 mL
- Dominant region: adnexal/pelvic
- Overall distribution: multifocal
- Dominant attenuation pattern: mixed solid-cystic and predominantly solid
- Heterogeneity: moderate
- Ascites: present
- Interpretive pattern: advanced omental and adnexal/pelvic disease pattern with ascites
- FIGO stage: III

<IMPRESSION>
Multifocal omental and adnexal/pelvic tumor burden totaling 38.8 mL, with ascites. Overall heterogeneous solid/mixed pattern is in keeping with FIGO stage III disease.
</IMPRESSION>

EXAMPLE 2:
KEY DATA:
- Sites: omental and adnexal/pelvic
- Number of tumor regions: 2
- Total tumor burden: 526.4 mL
- Dominant region: omental
- Overall distribution: extensive
- Dominant attenuation pattern: mixed solid-cystic
- Heterogeneity: marked
- Ascites: absent
- Interpretive pattern: advanced omental and adnexal/pelvic disease pattern
- FIGO stage: III

<IMPRESSION>
Extensive omental-dominant and adnexal/pelvic tumor burden totaling 526.4 mL, without ascites. Overall markedly heterogeneous mixed solid-cystic pattern is in keeping with FIGO stage III disease.
</IMPRESSION>

EXAMPLE 3:
KEY DATA:
- Sites: adnexal/pelvic only
- Number of tumor regions: 1
- Total tumor burden: 167.6 mL
- Dominant region: adnexal/pelvic
- Overall distribution: localized
- Dominant attenuation pattern: complex cystic
- Heterogeneity: mild
- Ascites: absent
- Interpretive pattern: localized adnexal/pelvic disease pattern
- FIGO stage: I

<IMPRESSION>
Localized adnexal/pelvic tumor burden totaling 167.6 mL, without ascites. Complex cystic morphology with mild heterogeneity is in keeping with FIGO stage I disease.
</IMPRESSION>

Now generate the IMPRESSION.

KEY DATA:
- Sites: {key_data['sites']}
- Number of tumor regions: {key_data['number_of_tumor_regions']}
- Total tumor burden: {key_data['total_tumor_burden']} mL
- Dominant region: {key_data['dominant_region']}
- Overall distribution: {key_data['overall_distribution']}
- Dominant attenuation pattern: {key_data['dominant_attenuation_pattern']}
- Heterogeneity: {key_data['heterogeneity']}
- Ascites: {key_data['ascites_status']}
- Interpretive pattern: {key_data['interpretive_pattern']}
- FIGO stage: {key_data['figo_stage']}

<IMPRESSION>"""

    @staticmethod
    def parse_output(text: str, tag_name: str) -> str:
        pattern = f"<{tag_name}>(.*?)</{tag_name}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        text = text.replace(f"<{tag_name}>", "").replace(f"</{tag_name}>", "")
        return text.strip()

    def generate_batch(self, prompts: list[str], is_findings: bool, batch_size: int) -> list[str]:
        params = self.findings_params if is_findings else self.impressions_params
        tag = "FINDINGS" if is_findings else "IMPRESSION"
        all_outputs = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Generating {tag}"):
            batch = prompts[i : i + batch_size]
            outputs = self.llm.generate(batch, params)
            for output in outputs:
                text = output.outputs[0].text.strip()
                all_outputs.append(self.parse_output(text, tag))
        return all_outputs

    def prepare_patients(
        self,
        metadata_csv: Path,
        repo_root: Path,
        only_patient_ids: set[str] | None = None,
        max_patients: int = 0,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
        rows = read_csv_dicts(metadata_csv)
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            patient_id = clean_text(row.get("patient_id", ""))
            if not patient_id:
                continue
            if only_patient_ids is not None and patient_id not in only_patient_ids:
                continue
            grouped[patient_id].append(row)

        patient_ids = sorted(grouped)
        if max_patients > 0:
            patient_ids = patient_ids[:max_patients]

        valid_patients: list[dict[str, Any]] = []
        tumor_feature_rows: list[dict[str, Any]] = []
        skipped_rows: list[dict[str, str]] = []

        for patient_id in tqdm(patient_ids, desc="Preparing"):
            patient_rows = grouped[patient_id]
            tumors_data = []

            for row in sorted(patient_rows, key=lambda r: int(r.get("label", 9))):
                try:
                    label = int(row.get("label", 9))
                    mask_path = ensure_path(row["tumor_mask_path"], repo_root, metadata_csv.parent)
                    if not mask_path.exists():
                        skipped_rows.append(
                            {
                                "patient_id": patient_id,
                                "label": str(label),
                                "reason": "missing tumor_mask_path",
                                "path": str(mask_path),
                            }
                        )
                        continue

                    radiomics = self.extract_shape_radiomics(mask_path, label=label)
                    organs_contact = [
                        item.strip()
                        for item in str(row.get("organs_contact", "")).split(",")
                        if item.strip()
                    ][:5]
                    patient_data = {
                        "location": LABEL_TO_LOCATION.get(label, "unknown"),
                        "volume_ml": float(row["volume_ml"]),
                        "mean_hu": float(row["mean_hu"]),
                        "p10_hu": float(row["p10_hu"]),
                        "p90_hu": float(row["p90_hu"]),
                        "organs_contact": organs_contact,
                    }
                    categorized = self.categorize_features(patient_data, radiomics)
                    tumors_data.append(
                        {
                            "label": label,
                            "location": patient_data["location"],
                            "volume_ml": patient_data["volume_ml"],
                            "patient_data": patient_data,
                            "categorized": categorized,
                        }
                    )
                    tumor_feature_rows.append(
                        {
                            "patient_id": patient_id,
                            "label": label,
                            "location": patient_data["location"],
                            "volume_ml": f"{patient_data['volume_ml']:.6f}",
                            "size_desc": categorized["size_desc"],
                            "size_cm": categorized["size_cm"],
                            "shape_desc": categorized["shape_desc"],
                            "density_primary": categorized["density_primary"],
                            "heterogeneity": categorized["heterogeneity"],
                            "has_low_density": categorized["has_low_density"],
                            "organs_contact": ",".join(organs_contact),
                        }
                    )
                except Exception as exc:
                    skipped_rows.append(
                        {
                            "patient_id": patient_id,
                            "label": str(row.get("label", "")),
                            "reason": f"{type(exc).__name__}: {exc}",
                            "path": row.get("tumor_mask_path", ""),
                        }
                    )

            if not tumors_data:
                continue

            first_row = patient_rows[0]
            figo = normalize_figo(first_row.get("figo_stage", "III"))
            ascites = normalize_ascites(first_row.get("ascites", "absent"))
            total_volume = sum(t["volume_ml"] for t in tumors_data)

            valid_patients.append(
                {
                    "patient_id": patient_id,
                    "volume_name": clean_text(first_row.get("volume_name", patient_id)) or patient_id,
                    "tumors_data": tumors_data,
                    "figo_stage": figo,
                    "ascites": ascites,
                    "total_volume_ml": total_volume,
                }
            )

        return valid_patients, tumor_feature_rows, skipped_rows

    def process_dataset(
        self,
        metadata_csv: Path,
        output_csv: Path,
        repo_root: Path,
        only_patient_ids: set[str] | None,
        max_patients: int,
        batch_size: int,
        save_every: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
        existing_rows: list[dict[str, Any]] = []
        processed: set[str] = set()
        if output_csv.exists():
            existing_rows = read_csv_dicts(output_csv)
            processed = {str(row["patient_id"]) for row in existing_rows}
            print(f"Resuming: {len(processed)} patients already in {output_csv}")

        valid_patients, tumor_feature_rows, skipped_rows = self.prepare_patients(
            metadata_csv=metadata_csv,
            repo_root=repo_root,
            only_patient_ids=only_patient_ids,
            max_patients=max_patients,
        )
        valid_patients = [p for p in valid_patients if p["patient_id"] not in processed]

        print(f"Prepared {len(valid_patients)} patients to generate")
        if not valid_patients:
            return existing_rows, tumor_feature_rows, skipped_rows

        findings_prompts = [
            self.build_findings_prompt(
                p["tumors_data"],
                p["figo_stage"],
                p["ascites"],
                p["total_volume_ml"],
            )
            for p in valid_patients
        ]
        findings_texts = self.generate_batch(findings_prompts, is_findings=True, batch_size=batch_size)
        for patient, findings in zip(valid_patients, findings_texts):
            patient["findings"] = findings

        impressions_prompts = [
            self.build_impressions_prompt(
                p["tumors_data"],
                p["figo_stage"],
                p["ascites"],
                p["total_volume_ml"],
            )
            for p in valid_patients
        ]
        impressions_texts = self.generate_batch(impressions_prompts, is_findings=False, batch_size=batch_size)

        results = existing_rows
        for patient, impressions in zip(valid_patients, impressions_texts):
            labels_str = ",".join(map(str, sorted(t["label"] for t in patient["tumors_data"])))
            row = {
                "prompt_version": "Promptgen_vers_6_structured_impression",
                "patient_id": patient["patient_id"],
                "volume_name": patient["volume_name"],
                "labels": labels_str,
                "findings": patient["findings"],
                "impressions": impressions,
                "combined_report": f"Findings: {patient['findings']} Impression: {impressions}",
                "figo_stage": patient["figo_stage"],
                "ascites": patient["ascites"],
                "total_volume_ml": f"{patient['total_volume_ml']:.1f}",
                "n_tumors": len(patient["tumors_data"]),
            }
            results.append(row)
            if len(results) % save_every == 0:
                write_csv_dicts(output_csv, results, FIELDNAMES)
                print(f"Saved {len(results)} rows")

        write_csv_dicts(output_csv, results, FIELDNAMES)
        return results, tumor_feature_rows, skipped_rows


def write_clip_reports(
    patient_rows: list[dict[str, Any]],
    out_dir: Path,
    train_ids: set[str],
    val_ids: set[str],
    test_ids: set[str],
) -> None:
    clip_rows = [
        {
            "VolumeName": row["volume_name"],
            "Findings_EN": row["findings"],
            "Impressions_EN": row["impressions"],
        }
        for row in sorted(patient_rows, key=lambda r: str(r["volume_name"]))
    ]
    write_csv_dicts(out_dir / "reports.csv", clip_rows, ["VolumeName", "Findings_EN", "Impressions_EN"])
    for split_name, split_ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        split_rows = [row for row in clip_rows if row["VolumeName"] in split_ids]
        write_csv_dicts(out_dir / f"reports_{split_name}.csv", split_rows, ["VolumeName", "Findings_EN", "Impressions_EN"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Promptgen v6 reports with structured, non-repetitive impressions.")
    parser.add_argument("--metadata_csv", default="data/metadata.csv")
    parser.add_argument("--out_dir", default="dataset/reports/Promptgen_vers_6")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--max_patients", type=int, default=0, help="Use a small value, e.g. 20, for a preview run.")
    parser.add_argument("--only_split_ids", action="store_true", help="Restrict generation to train/val/test JSON patients.")
    parser.add_argument("--train_json", default="dataset/unet_train_data_volumes.json")
    parser.add_argument("--val_json", default="dataset/unet_val_data_volumes.json")
    parser.add_argument("--test_json", default="dataset/unet_test_data_volumes.json")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    metadata_csv = ensure_path(args.metadata_csv, repo_root, repo_root)
    out_dir = ensure_path(args.out_dir, repo_root, repo_root)
    output_csv = out_dir / "patient_reports.csv"

    train_ids = load_ids_for_key(ensure_path(args.train_json, repo_root, repo_root), "training")
    val_ids = load_ids_for_key(ensure_path(args.val_json, repo_root, repo_root), "validation")
    test_ids = load_ids_for_key(ensure_path(args.test_json, repo_root, repo_root), "test")
    split_ids = train_ids | val_ids | test_ids
    only_patient_ids = split_ids if args.only_split_ids else None

    generator = StructuredImpressionReportGenerator(model_name=args.model_name)
    patient_rows, tumor_feature_rows, skipped_rows = generator.process_dataset(
        metadata_csv=metadata_csv,
        output_csv=output_csv,
        repo_root=repo_root,
        only_patient_ids=only_patient_ids,
        max_patients=args.max_patients,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )

    write_csv_dicts(
        out_dir / "patient_tumor_features.csv",
        tumor_feature_rows,
        [
            "patient_id",
            "label",
            "location",
            "volume_ml",
            "size_desc",
            "size_cm",
            "shape_desc",
            "density_primary",
            "heterogeneity",
            "has_low_density",
            "organs_contact",
        ],
    )
    if skipped_rows:
        write_csv_dicts(out_dir / "skipped_rows.csv", skipped_rows, ["patient_id", "label", "reason", "path"])

    write_clip_reports(patient_rows, out_dir, train_ids, val_ids, test_ids)

    summary = quality_metrics(patient_rows)
    (out_dir / "report_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved patient reports: {output_csv}")
    print(f"Saved CLIP reports and quality summary under: {out_dir}")
    if skipped_rows:
        reason_counts = Counter(row["reason"] for row in skipped_rows)
        print("Skipped rows:", dict(reason_counts.most_common()))


if __name__ == "__main__":
    main()
