#!/usr/bin/env python3
"""Batch-extract MedSyn CXR-BERT prompt features from text files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source_src", type=Path, required=True, help="MedSyn source/src directory.")
    ap.add_argument("--text_model_path", type=Path, required=True)
    ap.add_argument("--prompt_text_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--save_seq_len", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.source_src))
    from text_feature.modeling_cxrbert import CXRBertModel

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CXRBertModel.from_pretrained(str(args.text_model_path))
    old_embed = model.bert.embeddings.position_embeddings.weight.data
    old_len, hidden_size = old_embed.shape
    if args.max_seq_length > old_len:
        model.bert.embeddings.position_embeddings = nn.Embedding(args.max_seq_length, hidden_size)
        model.bert.embeddings.position_embeddings.weight.data[:old_len, :] = old_embed
        model.bert.embeddings.register_buffer(
            "position_ids", torch.arange(args.max_seq_length).expand((1, -1))
        )
        model.config.max_position_embeddings = args.max_seq_length

    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(args.text_model_path), trust_remote_code=True)

    prompt_files = sorted(args.prompt_text_dir.glob("*.txt"))
    if args.limit > 0:
        prompt_files = prompt_files[: args.limit]
    if not prompt_files:
        raise FileNotFoundError(f"No .txt prompts found in {args.prompt_text_dir}")

    for path in tqdm(prompt_files, desc="MedSyn prompt features"):
        text = path.read_text().strip()
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            feature = model(**encoded)
        feature_np = feature.hidden_states[-1][:, : args.save_seq_len, :].detach().cpu().numpy()
        np.save(args.out_dir / f"{path.stem}.npy", feature_np.astype(np.float32))

    print(f"Wrote {len(prompt_files)} prompt features to {args.out_dir}")


if __name__ == "__main__":
    main()
