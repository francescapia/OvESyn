#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dedup_on", default="patient_id", choices=["patient_id", "ct_path", "none"])
    args = ap.parse_args()

    tr = pd.read_csv(args.train_csv).fillna("")
    va = pd.read_csv(args.val_csv).fillna("")

    # controlla schema
    if list(tr.columns) != list(va.columns):
        missing_in_va = [c for c in tr.columns if c not in va.columns]
        missing_in_tr = [c for c in va.columns if c not in tr.columns]
        raise SystemExit(
            "Le colonne non coincidono.\n"
            f"Missing in val: {missing_in_va}\n"
            f"Missing in train: {missing_in_tr}\n"
            "Soluzione: allinea le colonne prima (stesso ordine e stessi nomi)."
        )

    df = pd.concat([tr, va], ignore_index=True)

    if args.dedup_on != "none":
        key = args.dedup_on
        if key not in df.columns:
            raise SystemExit(f"Colonna '{key}' non trovata nel CSV, disponibili: {list(df.columns)}")
        # tiene la prima occorrenza (train “vince” su val perché è concatenato prima)
        df = df.drop_duplicates(subset=[key], keep="first")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Saved:", out)
    print("Train rows:", len(tr), "Val rows:", len(va), "Merged rows:", len(df))

if __name__ == "__main__":
    main()
