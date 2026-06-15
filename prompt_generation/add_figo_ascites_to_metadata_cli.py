#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_FIGO_COL = "Ovarian/Peritoneum/Fallopian Tube Cancer FIGO Staging  (based on clinical and pathological findings at the diagnosis)"
DEFAULT_ASCITES_COL = "Ascites?y/n"
DEFAULT_ID_CANDIDATES = [
    "Record ID",
    "VolumeName",
    "patient_id",
    "PatientID",
    "ID",
    "id",
    "Paziente",
    "Codice",
    "code",
]


def normalize_figo(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        mapping = {1: "I", 2: "II", 3: "III", 4: "IV"}
        return mapping.get(int(value), "")
    text = str(value).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+(\.0+)?", text):
        mapping = {1: "I", 2: "II", 3: "III", 4: "IV"}
        return mapping.get(int(float(text)), "")
    text_up = text.upper()
    for roman in ["IV", "III", "II", "I"]:
        if roman in text_up:
            return roman
    return ""


def normalize_ascites(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "present", "positivo", "si", "sì"}:
        return "present"
    if text in {"no", "n", "false", "0", "absent", "negativo"}:
        return "absent"
    return ""


def normalize_patient_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(value).strip())


def find_id_column(df: pd.DataFrame) -> str:
    for col in DEFAULT_ID_CANDIDATES:
        if col in df.columns:
            return col
    pat = re.compile(r"^IEO[\s\-]?\d+[\d\-]*$", re.IGNORECASE)
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(200)
        if not sample.empty and sample.str.match(pat).any():
            return col
    raise RuntimeError("Could not infer patient ID column from clinical spreadsheet.")


def find_column(df: pd.DataFrame, requested: str, aliases: list[str], purpose: str) -> str:
    if requested in df.columns:
        return requested

    lowered = {str(col).strip().lower(): col for col in df.columns}
    if requested.strip().lower() in lowered:
        return lowered[requested.strip().lower()]

    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.strip().lower() in lowered:
            return lowered[alias.strip().lower()]

    if purpose == "ascites":
        candidates = [
            col
            for col in df.columns
            if str(col).lower().startswith("ascites?") and "recurrence" not in str(col).lower()
        ]
        if candidates:
            # Prefer the first binary column when pandas mangles duplicate names to Ascites? / Ascites?.1
            candidates = sorted(candidates, key=lambda c: (".1" in str(c), str(c)))
            return candidates[0]

    raise RuntimeError(
        f"Could not infer {purpose} column from clinical spreadsheet. Requested='{requested}'. "
        f"Available columns include: {list(df.columns[:20])} ..."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Add FIGO stage and ascites information to prompt-generation metadata.")
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--clinical_xlsx", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--figo_col", default=DEFAULT_FIGO_COL)
    parser.add_argument("--ascites_col", default=DEFAULT_ASCITES_COL)
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_csv).fillna("")
    clinical = pd.read_excel(args.clinical_xlsx, engine="openpyxl")
    id_col = find_id_column(clinical)
    figo_col = find_column(
        clinical,
        requested=args.figo_col,
        aliases=[DEFAULT_FIGO_COL, "dc_figo_staging"],
        purpose="figo",
    )
    ascites_col = find_column(
        clinical,
        requested=args.ascites_col,
        aliases=[DEFAULT_ASCITES_COL, "Ascites?", "Ascites?.1", "dc_oc_ascites"],
        purpose="ascites",
    )

    clinical_small = clinical[[id_col, figo_col, ascites_col]].copy()
    clinical_small[id_col] = clinical_small[id_col].apply(normalize_patient_id)
    clinical_small["figo_stage"] = clinical_small[figo_col].apply(normalize_figo)
    clinical_small["ascites"] = clinical_small[ascites_col].apply(normalize_ascites)
    clinical_small = clinical_small[
        (clinical_small[id_col].astype(str).str.strip() != "")
        & (
            (clinical_small["figo_stage"].astype(str).str.strip() != "")
            | (clinical_small["ascites"].astype(str).str.strip() != "")
        )
    ]
    clinical_small = clinical_small.drop_duplicates(subset=[id_col], keep="first")

    merged = metadata.copy()
    merged["patient_id"] = merged["patient_id"].apply(normalize_patient_id)
    merged = merged.merge(
        clinical_small[[id_col, "figo_stage", "ascites"]].rename(columns={id_col: "patient_id"}),
        on="patient_id",
        how="left",
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"Saved metadata with clinical fields: {output_csv}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()
