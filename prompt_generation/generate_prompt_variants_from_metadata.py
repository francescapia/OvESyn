#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from prompt_generation.generate_captions_local_v3 import LABEL_TO_LOCATION, DualReportGenerator


VARIANT_ORDER = ["P0", "P1", "P2", "P3"]


def load_ids_for_key(json_path: Path, key: str) -> set[str]:
    data = json.loads(json_path.read_text())
    return {Path(item["image"]).parent.name for item in data.get(key, [])}


def clean_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def normalize_figo(value: object) -> str:
    text = clean_text(value)
    return text or "III"


def normalize_ascites(value: object) -> str:
    text = clean_text(str(value).lower())
    if text in {"present", "yes", "y", "true", "1"}:
        return "present"
    if text in {"absent", "no", "n", "false", "0"}:
        return "absent"
    return "absent"


def ensure_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def summarize_variant(prompt_id: str) -> dict[str, bool]:
    return {
        "include_intensity": prompt_id in {"P1", "P2", "P3"},
        "include_location": prompt_id in {"P2", "P3"},
        "include_size": prompt_id in {"P2", "P3"},
        "include_volume": prompt_id in {"P2", "P3"},
        "include_morphology": prompt_id in {"P2", "P3"},
        "include_hu_details": prompt_id in {"P1", "P2", "P3"},
        "include_adjacency": prompt_id == "P3",
    }


def findings_example(prompt_id: str) -> str:
    base_t1 = "Tumor 1 (omental):"
    base_t2 = "Tumor 2 (pelvic/ovarian):"

    if prompt_id == "P0":
        return """<FINDINGS>
Combined tumor burden is 38.8 mL. Ascites is present. Findings are consistent with FIGO Stage III disease.
</FINDINGS>"""
    if prompt_id == "P1":
        return """<FINDINGS>
An omental lesion demonstrates predominantly solid attenuation with moderate heterogeneity. A separate pelvic/ovarian lesion shows mixed solid and cystic components with mild heterogeneity. Combined tumor burden is 38.8 mL. Ascites is present. Findings are consistent with FIGO Stage III disease.
</FINDINGS>"""
    if prompt_id == "P2":
        return """<FINDINGS>
An irregular multilobulated omental mass measuring 14.9 cm with volume 31.2 mL demonstrates predominantly solid attenuation with moderate heterogeneity. A separate lobulated pelvic/ovarian mass measuring 7.6 cm with volume 7.6 mL shows mixed solid and cystic components with mild heterogeneity. Combined tumor burden is 38.8 mL. Ascites is present. Findings are consistent with FIGO Stage III disease.
</FINDINGS>"""
    return """<FINDINGS>
An irregular multilobulated omental mass measuring 14.9 cm with volume 31.2 mL demonstrates predominantly solid attenuation with moderate heterogeneity. The mass abuts small bowel and colon. A separate lobulated pelvic/ovarian mass measuring 7.6 cm with volume 7.6 mL shows mixed solid and cystic components with mild heterogeneity, abutting bladder. Combined tumor burden is 38.8 mL. Ascites is present. Findings are consistent with FIGO Stage III disease.
</FINDINGS>"""


def impression_example(prompt_id: str) -> str:
    if prompt_id == "P0":
        return """<IMPRESSION>
Combined tumor burden of 38.8 mL. Findings consistent with FIGO Stage III disease. Ascites is present.
</IMPRESSION>"""
    if prompt_id == "P1":
        return """<IMPRESSION>
Two lesions demonstrate mixed solid and cystic or predominantly solid attenuation with heterogeneous appearance. Combined burden is 38.8 mL. Findings consistent with FIGO Stage III disease.
</IMPRESSION>"""
    if prompt_id == "P2":
        return """<IMPRESSION>
Multifocal masses in omentum and pelvis/ovaries with combined burden of 38.8 mL. Features include heterogeneous attenuation and lesion morphology. Findings consistent with FIGO Stage III disease.
</IMPRESSION>"""
    return """<IMPRESSION>
Multifocal masses in omentum and pelvis/ovaries with combined burden of 38.8 mL. Findings consistent with FIGO Stage III disease. Features include heterogeneous attenuation and contact with multiple organs.
</IMPRESSION>"""


class VariantPromptReportGenerator(DualReportGenerator):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct"):
        super().__init__(model_name=model_name)

    def build_tumor_descriptions(self, tumors_data: list[dict[str, Any]], prompt_id: str) -> str:
        cfg = summarize_variant(prompt_id)
        tumor_descriptions = []

        for i, tumor in enumerate(tumors_data, 1):
            title = f"Tumor {i}"
            if cfg["include_location"]:
                title += f" ({tumor['location']})"
            desc_lines = [title + ":"]

            if cfg["include_size"]:
                desc_lines.append(
                    f"- Size: {tumor['categorized']['size_desc']} measuring {tumor['categorized']['size_cm']}"
                )
            if cfg["include_volume"]:
                desc_lines.append(f"- Individual volume: {tumor['volume_ml']:.1f} mL")
            if cfg["include_morphology"]:
                desc_lines.append(f"- Morphology: {tumor['categorized']['shape_desc']}")
            if cfg["include_intensity"]:
                desc_lines.append(
                    f"- Attenuation: {tumor['categorized']['density_primary']}, {tumor['categorized']['heterogeneity']}"
                )
                if cfg["include_hu_details"]:
                    desc_lines.append(
                        f"- Mean HU: {tumor['categorized']['mean_hu']:.0f} "
                        f"(range {tumor['patient_data']['p10_hu']:.0f}-{tumor['patient_data']['p90_hu']:.0f})"
                    )
                if tumor["categorized"]["has_low_density"]:
                    desc_lines.append("- Low attenuation areas present")
            if cfg["include_adjacency"]:
                organs = ", ".join(tumor["patient_data"]["organs_contact"][:4])
                if organs:
                    desc_lines.append(f"- Adjacent organs: {organs}")

            tumor_descriptions.append("\n".join(desc_lines))

        return "\n\n".join(tumor_descriptions)

    def build_findings_prompt_variant(self, tumors_data, figo_stage, ascites, total_volume_ml, prompt_id: str):
        tumors_text = self.build_tumor_descriptions(tumors_data, prompt_id)
        if ascites.lower() == "present":
            ascites_instruction = "State: 'Ascites is present.'"
        else:
            ascites_instruction = "State: 'No ascites is identified.'"

        extra_rule = ""
        if prompt_id == "P0":
            extra_rule = "\n7. Do not mention lesion site, size, morphology, attenuation, or organ contact unless explicitly present in PATIENT DATA."
        elif prompt_id == "P1":
            extra_rule = "\n7. Do not mention lesion site, size, morphology, individual lesion volume, or organ contact unless explicitly present in PATIENT DATA."
        elif prompt_id == "P2":
            extra_rule = "\n7. Do not mention organ contact or adjacency unless explicitly present in PATIENT DATA."

        prompt = f"""You are a radiologist writing objective CT findings. Output ONLY the findings text within <FINDINGS></FINDINGS> tags.

CRITICAL RULES - NO EXCEPTIONS:
1. NEVER use: "invades", "invasion", "metastatic", "malignancy", "peritoneal implants"
2. For organ contact, use ONLY: "abuts [organ]" (do not add "without invasion")
3. Ascites: {ascites_instruction} (NEVER "mild/moderate/severe")
4. FIGO: State only "Findings are consistent with FIGO Stage {figo_stage} disease." (do not elaborate)
5. Use ONLY information from PATIENT DATA below - no additional clinical interpretation
6. Report individual volumes AND combined total{extra_rule}

EXAMPLE:
{findings_example(prompt_id)}

PATIENT DATA:
{tumors_text}

Combined tumor burden: {total_volume_ml:.1f} mL
FIGO stage: {figo_stage}
Ascites: {ascites}

<FINDINGS>"""
        return prompt

    def build_impressions_prompt_variant(self, findings_text, tumors_data, figo_stage, ascites, total_volume_ml, prompt_id: str):
        cfg = summarize_variant(prompt_id)
        locations = [t["location"] for t in tumors_data]
        if cfg["include_location"] and locations:
            location_desc = "omentum and pelvis/ovaries" if len(locations) > 1 else locations[0]
        else:
            location_desc = "not provided"

        extra_rule = ""
        if prompt_id == "P0":
            extra_rule = "\n6. Do not mention lesion size, site, morphology, attenuation, or organ contact unless they appear in FINDINGS."
        elif prompt_id == "P1":
            extra_rule = "\n6. Do not mention lesion site, size, morphology, individual lesion volume, or organ contact unless they appear in FINDINGS."
        elif prompt_id == "P2":
            extra_rule = "\n6. Do not mention organ contact or adjacency unless they appear in FINDINGS."

        prompt = f"""You are a radiologist writing the impression. Output ONLY within <IMPRESSION></IMPRESSION> tags.

CRITICAL RULES:
1. Use descriptive terms: "mass", "masses" (NEVER "malignancy", "carcinoma", "metastatic disease")
2. State FIGO correlation: "Findings consistent with FIGO Stage {figo_stage} disease."
3. NEVER add: "invasion", "peritoneal spread", "lymph nodes", "metastases"
4. Describe only what's in FINDINGS (do not invent new imaging features)
5. Maximum 40 words, 2-3 sentences{extra_rule}

EXAMPLE:
{impression_example(prompt_id)}

FINDINGS:
{findings_text}

KEY DATA:
- Sites: {location_desc}
- Total burden: {total_volume_ml:.1f} mL
- FIGO: {figo_stage}
- Ascites: {ascites}

<IMPRESSION>"""
        return prompt

    def prepare_patients(self, metadata_csv: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        df = pd.read_csv(metadata_csv)
        print(f"Loaded {len(df)} rows")
        grouped = df.groupby("patient_id")

        valid_patients: list[dict[str, Any]] = []
        tumor_feature_rows: list[dict[str, Any]] = []

        for patient_id in tqdm(grouped.groups.keys(), desc="Preparing"):
            try:
                patient_rows = grouped.get_group(patient_id)
                tumors_data = []

                for _, row in patient_rows.iterrows():
                    label = int(row.get("label", 9))
                    ct_path = ensure_path(str(row["ct_path"]), metadata_csv.parent)
                    mask_path = ensure_path(str(row["tumor_mask_path"]), metadata_csv.parent)

                    if not ct_path.exists() or not mask_path.exists():
                        continue

                    radiomics = self.extract_radiomics(ct_path, mask_path, label=label)
                    patient_data = {
                        "location": LABEL_TO_LOCATION.get(label, "unknown"),
                        "volume_ml": float(row["volume_ml"]),
                        "mean_hu": float(row["mean_hu"]),
                        "p10_hu": float(row["p10_hu"]),
                        "p90_hu": float(row["p90_hu"]),
                        "organs_contact": [x.strip() for x in str(row["organs_contact"]).split(",") if x.strip()][:5],
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
                            "volume_ml": patient_data["volume_ml"],
                            "size_desc": categorized["size_desc"],
                            "size_cm": categorized["size_cm"],
                            "shape_desc": categorized["shape_desc"],
                            "density_primary": categorized["density_primary"],
                            "heterogeneity": categorized["heterogeneity"],
                            "has_low_density": categorized["has_low_density"],
                            "organs_contact": ",".join(patient_data["organs_contact"]),
                        }
                    )

                if not tumors_data:
                    continue

                first_row = patient_rows.iloc[0]
                figo = normalize_figo(first_row.get("figo_stage", "III"))
                ascites = normalize_ascites(first_row.get("ascites", "absent"))
                total_volume = sum(t["volume_ml"] for t in tumors_data)

                valid_patients.append(
                    {
                        "patient_id": patient_id,
                        "tumors_data": tumors_data,
                        "figo_stage": figo,
                        "ascites": ascites,
                        "total_volume_ml": total_volume,
                        "volume_name": first_row.get("volume_name", patient_id),
                    }
                )
            except Exception as exc:
                print(f"❌ {patient_id}: {exc}")
                continue

        print(f"✓ {len(valid_patients)} valid patients")
        return valid_patients, tumor_feature_rows

    def generate_variant_reports(self, valid_patients: list[dict[str, Any]], prompt_id: str, batch_size: int) -> list[dict[str, Any]]:
        print(f"\n=== Building Findings prompts for {prompt_id} ===")
        findings_prompts = [
            self.build_findings_prompt_variant(
                p["tumors_data"],
                p["figo_stage"],
                p["ascites"],
                p["total_volume_ml"],
                prompt_id,
            )
            for p in valid_patients
        ]

        print(f"\n=== Generating Findings for {prompt_id} ===")
        findings_texts = self.generate_batch(findings_prompts, is_findings=True, batch_size=batch_size)

        enriched_patients = []
        for patient, findings in zip(valid_patients, findings_texts):
            item = dict(patient)
            item["findings"] = findings
            enriched_patients.append(item)

        print(f"\n=== Building Impressions prompts for {prompt_id} ===")
        impressions_prompts = [
            self.build_impressions_prompt_variant(
                p["findings"],
                p["tumors_data"],
                p["figo_stage"],
                p["ascites"],
                p["total_volume_ml"],
                prompt_id,
            )
            for p in enriched_patients
        ]

        print(f"\n=== Generating Impressions for {prompt_id} ===")
        impressions_texts = self.generate_batch(impressions_prompts, is_findings=False, batch_size=batch_size)

        results = []
        for patient, impressions in zip(enriched_patients, impressions_texts):
            labels_str = ",".join(map(str, sorted([t["label"] for t in patient["tumors_data"]])))
            results.append(
                {
                    "prompt_id": prompt_id,
                    "patient_id": patient["patient_id"],
                    "volume_name": patient["volume_name"],
                    "labels": labels_str,
                    "findings": patient["findings"],
                    "impressions": impressions,
                    "combined_report": f"Findings: {patient['findings']} Impression: {impressions}",
                    "figo_stage": patient["figo_stage"],
                    "ascites": patient["ascites"],
                    "total_volume_ml": patient["total_volume_ml"],
                    "n_tumors": len(patient["tumors_data"]),
                }
            )
        return results


def write_clip_reports(rows: list[dict[str, Any]], out_dir: Path, train_ids: set[str], val_ids: set[str], test_ids: set[str]) -> None:
    patient_df = pd.DataFrame(rows).sort_values("patient_id")
    clip_df = patient_df.rename(
        columns={
            "volume_name": "VolumeName",
            "findings": "Findings_EN",
            "impressions": "Impressions_EN",
        }
    )[["VolumeName", "Findings_EN", "Impressions_EN"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    patient_df.to_csv(out_dir / "patient_reports.csv", index=False)
    clip_df.to_csv(out_dir / "reports.csv", index=False)
    clip_df[clip_df["VolumeName"].isin(train_ids)].to_csv(out_dir / "reports_train.csv", index=False)
    clip_df[clip_df["VolumeName"].isin(val_ids)].to_csv(out_dir / "reports_val.csv", index=False)
    clip_df[clip_df["VolumeName"].isin(test_ids)].to_csv(out_dir / "reports_test.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prompt-paper prompt variants using the original Qwen + PyRadiomics pipeline.")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--variants_json", default="prompt_paper/set1_variants.json")
    parser.add_argument("--train_json", default="dataset/unet_train_data_volumes.json")
    parser.add_argument("--val_json", default="dataset/unet_val_data_volumes.json")
    parser.add_argument("--test_json", default="dataset/unet_test_data_volumes.json")
    parser.add_argument("--out_root", default="dataset/prompt_paper/set1")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    metadata_csv = ensure_path(args.metadata_csv, repo_root)
    out_root = ensure_path(args.out_root, repo_root)
    variants = json.loads((repo_root / args.variants_json).read_text())
    variant_ids = [v["id"] for v in variants if v["id"] in VARIANT_ORDER]

    train_ids = load_ids_for_key(repo_root / args.train_json, "training")
    val_ids = load_ids_for_key(repo_root / args.val_json, "validation")
    test_ids = load_ids_for_key(repo_root / args.test_json, "test")

    generator = VariantPromptReportGenerator(model_name=args.model_name)
    valid_patients, tumor_feature_rows = generator.prepare_patients(metadata_csv)

    all_rows = []
    for prompt_id in variant_ids:
        rows = generator.generate_variant_reports(valid_patients, prompt_id=prompt_id, batch_size=args.batch_size)
        write_clip_reports(rows, out_root / prompt_id / "reports", train_ids, val_ids, test_ids)
        all_rows.extend(rows)

    pd.DataFrame(tumor_feature_rows).sort_values(["patient_id", "label"]).to_csv(out_root / "patient_tumor_features.csv", index=False)
    pd.DataFrame(all_rows).sort_values(["prompt_id", "patient_id"]).to_csv(out_root / "patient_reports_all_variants.csv", index=False)

    print(f"\n✓ Saved all variants under ./{out_root.relative_to(repo_root).as_posix()}")


if __name__ == "__main__":
    main()
