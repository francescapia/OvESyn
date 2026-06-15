import argparse
import pandas as pd

def build_context_row(figo_stage, ascites, total_volume_ml):
    # Standardizza ascites
    asc = str(ascites).strip().lower()
    if asc in ["present", "yes", "y", "1", "true"]:
        asc_txt = "present"
    elif asc in ["absent", "no", "n", "0", "false"]:
        asc_txt = "absent"
    else:
        asc_txt = str(ascites)

    figo = str(figo_stage).strip()
    try:
        vol = float(total_volume_ml)
        vol_txt = f"{vol:.1f} mL"
    except Exception:
        vol_txt = str(total_volume_ml)

    return f"Clinical context: FIGO stage {figo}; Ascites {asc_txt}; Total tumor burden {vol_txt}.\n\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="radiological_reports_final.csv")
    ap.add_argument("--output_csv", required=True, help="reports.csv for CLIP3D")
    ap.add_argument("--prepend_context", action="store_true",
                    help="If set, prepend a standardized clinical context line to Findings_EN")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    # Basic sanity checks
    required_cols = ["patient_id", "volume_name", "findings", "impressions"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")

    out = pd.DataFrame()
    out["VolumeName"] = df["volume_name"].fillna(df["patient_id"]).astype(str)

    if args.prepend_context:
        ctx = df.apply(
            lambda r: build_context_row(r.get("figo_stage", ""), r.get("ascites", ""), r.get("total_volume_ml", "")),
            axis=1
        )
        out["Findings_EN"] = (ctx + df["findings"].fillna("").astype(str)).str.strip()
    else:
        out["Findings_EN"] = df["findings"].fillna("").astype(str).str.strip()

    out["Impressions_EN"] = df["impressions"].fillna("").astype(str).str.strip()

    out.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out)} rows to: {args.output_csv}")

if __name__ == "__main__":
    main()
