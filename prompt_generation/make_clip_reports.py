#!/usr/bin/env python3
import argparse
import re
import pandas as pd

LABEL_TO_LOCATION = {1: "omental", 9: "pelvic/ovarian"}

CUTOFF_PATTERNS = [
    r"\n\s*Note\s*:",
    r"\n\s*Here['’]s the revised version\s*:",
    r"\n\s*I have taken out",
    r"\n\s*This response follows",
    r"\n\s*Let me know if",
]

def strip_llm_meta(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text)

    t = re.sub(r"^\s*FINDINGS\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^\s*IMPRESSION\s*:\s*", "", t, flags=re.IGNORECASE)

    for pat in CUTOFF_PATTERNS:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            t = t[:m.start()]
            break

    m = re.search(r"\bNote\s*:", t, flags=re.IGNORECASE)
    if m:
        t = t[:m.start()]

    t = t.replace("\r\n", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    return t.strip()

def fmt_volume_ml(v) -> str:
    try:
        if pd.isna(v):
            return "NA"
        return f"{float(v):.1f}"
    except Exception:
        return "NA"

def fmt_figo(figo) -> str:
    if pd.isna(figo) or str(figo).strip() == "":
        return "NA"
    return str(figo).strip()

def fmt_ascites(a) -> str:
    if pd.isna(a) or str(a).strip() == "":
        return "NA"
    return str(a).strip()

def merge_patient_group(g: pd.DataFrame) -> pd.Series:
    g = g.copy()
    g["label"] = pd.to_numeric(g.get("label"), errors="coerce")
    g = g.sort_values("label")  # 1 poi 9

    patient_id = str(g.iloc[0].get("patient_id", "")).strip()
    figo = fmt_figo(g.iloc[0].get("figo_stage", "NA"))
    asc  = fmt_ascites(g.iloc[0].get("ascites", "NA"))

    # Header patient-level (NO volume qui)
    context_header = f"Clinical context: FIGO stage {figo}; Ascites {asc}."

    findings_parts = []
    impressions_parts = []

    for _, r in g.iterrows():
        label = int(r["label"]) if pd.notna(r["label"]) else -1
        loc = LABEL_TO_LOCATION.get(label, "unknown")
        vol = fmt_volume_ml(r.get("volume_ml", "NA"))

        # Prefisso con volume per label
        if label != -1:
            prefix = f"{loc} tumor (label {label}, volume {vol} mL): "
        else:
            prefix = f"tumor (volume {vol} mL): "

        f = strip_llm_meta(r.get("findings", ""))
        i = strip_llm_meta(r.get("impressions", ""))

        # Fallback da combined_report se serve
        if not f:
            cr = strip_llm_meta(r.get("combined_report", ""))
            if "Impression:" in cr:
                f = cr.split("Impression:")[0].replace("Findings:", "").strip()
        if not i:
            cr = strip_llm_meta(r.get("combined_report", ""))
            if "Impression:" in cr:
                i = cr.split("Impression:", 1)[1].strip()

        if f:
            findings_parts.append(prefix + f)
        if i:
            impressions_parts.append(prefix + i)

    findings_text = "\n\n".join(findings_parts).strip()
    impressions_text = "\n\n".join(impressions_parts).strip()

    # Findings con header standardizzato in cima
    if findings_text:
        findings_with_header = (context_header + "\n\n" + findings_text).strip()
    else:
        findings_with_header = context_header

    return pd.Series({
        "VolumeName": patient_id,
        "Findings_EN": findings_with_header,
        "Impressions_EN": impressions_text,
    })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = {"patient_id", "label", "volume_ml"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")

    out = df.groupby("patient_id", as_index=False).apply(merge_patient_group).reset_index(drop=True)

    out.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out)} rows -> {args.output_csv}")
    print("Columns:", list(out.columns))
    print("\nSample:")
    print(out.head(1).to_string(index=False))

if __name__ == "__main__":
    main()

