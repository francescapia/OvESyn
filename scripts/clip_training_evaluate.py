#!/usr/bin/env python3
"""
CLIP3D evaluation (coerente con il training in main_clip_training.py e Clip_Training_Script.py)

Metriche:
1) Loss (come eval_epoch_loss)
2) Retrieval globale (I2T e T2I) + CLIP score diagonale
3) Linear probing (Accuracy + AUC) usando split coerente col training:
   - train_probing = training + validation
   - test_probing  = test (da config.test_data_list; richiesto quando il probing è attivo)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.clip_training.utils.util import mkdir, set_seed, load_config_file  # :contentReference[oaicite:2]{index=2}
from core.cfg_helper import model_cfg_bank  # :contentReference[oaicite:3]{index=3}
from core.models.common.get_model import get_model  # :contentReference[oaicite:4]{index=4}
import monai
from monai.data import DataLoader
from monai.transforms import Compose

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import re

from monai.transforms import CopyItemsd


# -----------------------------
# Label extraction from text (NO tumor_type)
# -----------------------------
FIGO_RE = re.compile(r"\bFIGO\s*Stage\s*(III|IV)\b", re.IGNORECASE)
ASCITES_POS_RE = re.compile(r"\bAscites\s+is\s+present\b", re.IGNORECASE)
ASCITES_NEG_RE = re.compile(
    r"\bNo\s+ascites\s+is\s+identified\b|\bAscites\s+is\s+absent\b|\bAbsence\s+of\s+ascites\b|\bAbsent\s+ascites\b",
    re.IGNORECASE,
)
HET_RE = re.compile(r"\b(mild|moderate|marked)\s+heterogeneity\b", re.IGNORECASE)
SHAPE_WORDS = ["ovoid", "lobulated", "elongated", "flattened"]

HET_MAP = {"mild": 0, "moderate": 1, "marked": 2}

def _build_label_text(findings: str, impressions: str) -> str:
    f = "" if findings is None or (isinstance(findings, float) and np.isnan(findings)) else str(findings)
    im = "" if impressions is None or (isinstance(impressions, float) and np.isnan(impressions)) else str(impressions)
    return (f.strip() + "\n" + im.strip()).strip()

def extract_figo_stage(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    m = FIGO_RE.search(text)
    if not m:
        return None
    roman = m.group(1).upper()
    return 3 if roman == "III" else 4

def extract_ascites(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    if ASCITES_POS_RE.search(text):
        return 1
    if ASCITES_NEG_RE.search(text):
        return 0
    return None

def extract_heterogeneity(text: str):
    if not isinstance(text, str) or not text.strip():
        return None
    m = HET_RE.search(text)
    if not m:
        return None
    return m.group(1).lower()

def extract_shape_flags(text: str):
    out = {s: None for s in SHAPE_WORDS}
    if not isinstance(text, str) or not text.strip():
        return out
    t = text.lower()
    for s in SHAPE_WORDS:
        out[s] = 1 if re.search(rf"\b{s}\b", t) else 0
    return out

def debug_lora_devices(model, max_print=10, prefix=""):
    # se DataParallel, guarda il modulo interno
    m = model.module if hasattr(model, "module") else model

    lora_params = []
    for name, p in m.named_parameters():
        if "lora" in name.lower():  # prende lora_A, lora_B, ecc.
            lora_params.append((name, p.device, float(p.detach().abs().mean().cpu())))

    print(f"{prefix}N lora params: {len(lora_params)}")
    if not lora_params:
        return

    # conta quanti stanno su ciascun device
    from collections import Counter
    devs = Counter([str(d) for _, d, _ in lora_params])
    print(f"{prefix}LoRA devices: {dict(devs)}")

    # mostra i primi max_print
    print(f"{prefix}First {min(max_print, len(lora_params))}:")
    for t in lora_params[:max_print]:
        print(prefix + str(t))


# ---------------------------------------------------------------------
# Repo path setup (come negli script training)
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


LEGACY_DATA_PREFIX = "data/private_ct/"
PREPROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "private_ct_preprocessed"
PROCESSED_CT_NAME = "ct_preprocessed.nii.gz"
RAW_CT_NAME = "ct.nii.gz"


def resolve_repo_path(path_str: str | os.PathLike) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    repo_candidate = PROJECT_ROOT / path
    if repo_candidate.exists():
        return repo_candidate

    return path


def resolve_volume_path(image_path: str | os.PathLike) -> Path | None:
    raw_path = Path(image_path)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
        if raw_path.name == PROCESSED_CT_NAME:
            candidates.append(raw_path.with_name(RAW_CT_NAME))
    else:
        candidates.append(PROJECT_ROOT / raw_path)
        candidates.append(Path.cwd() / raw_path)

    normalized = raw_path.as_posix()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.endswith(PROCESSED_CT_NAME):
        candidates.append(PROJECT_ROOT / normalized.replace(PROCESSED_CT_NAME, RAW_CT_NAME))
    if normalized.startswith(LEGACY_DATA_PREFIX):
        suffix = normalized[len(LEGACY_DATA_PREFIX):]
        candidates.append(PREPROCESSED_DATA_ROOT / suffix)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve(strict=False)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return None



# LoRA
try:
    from lora_layers import apply_lora_to_model
except Exception:
    apply_lora_to_model = None


# ============================================================================
# Logger
# ============================================================================
def get_logger(name: str = "clip_eval") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# ============================================================================
# Checkpoint helpers
# ============================================================================
def _extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """Supporta:
    - state_dict nudo (OrderedDict/dict di tensori)
    - dict con chiavi tipo 'state_dict' / 'model_state_dict' / 'model' / 'net'
    """
    if isinstance(ckpt_obj, (dict, OrderedDict)):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if isinstance(ckpt_obj, dict) and key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        # già state_dict nudo
        if all(isinstance(k, str) for k in ckpt_obj.keys()):
            return dict(ckpt_obj)
    raise ValueError("Formato checkpoint non riconosciuto (atteso state_dict nudo o dict con state_dict).")


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def _looks_like_lora(sd: Dict[str, torch.Tensor]) -> bool:
    # Nel tuo caso LoRA usa wrapper con .linear.* e .lora.*
    keys = sd.keys()
    return any(".lora." in k for k in keys) or any(".linear." in k for k in keys)


def _apply_lora_wrappers_if_needed(config, model, sd, logger):
    if not _looks_like_lora(sd):
        return model

    logger.info("Checkpoint LoRA rilevato (.linear/.lora): applico apply_lora_to_model() prima del load.")
    if apply_lora_to_model is None:
        raise ImportError("Non riesco a importare apply_lora_to_model da lora_layers (PYTHONPATH?).")

    if not hasattr(config, "lora"):
        raise ValueError("Checkpoint LoRA ma config.lora non presente: usa la YAML del training LoRA.")

    lora_cfg = config.lora
    target_modules = lora_cfg["target_modules"] if isinstance(lora_cfg, dict) else lora_cfg.target_modules
    r = lora_cfg["r"] if isinstance(lora_cfg, dict) else lora_cfg.r
    alpha = lora_cfg["alpha"] if isinstance(lora_cfg, dict) else lora_cfg.alpha
    dropout = lora_cfg["dropout"] if isinstance(lora_cfg, dict) else lora_cfg.dropout

    return apply_lora_to_model(
        model=model,
        target_modules=target_modules,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        verbose=False,
    )


# ============================================================================
# Data loading (coerente con main_clip_training.py)
# ============================================================================
def load_filenames(data_list_path: str, split: str = "training") -> list:
    import json
    data_list_path = resolve_repo_path(data_list_path)
    with open(data_list_path, "r") as file:
        json_data = json.load(file)

    if split not in json_data:
        # fallback utile per JSON “monosplit”
        if split == "validation" and "training" in json_data:
            split = "training"
        elif split == "training" and "validation" in json_data:
            split = "validation"
        else:
            raise KeyError(f"Split '{split}' not found in {data_list_path}. Available keys: {list(json_data.keys())}")

    return [_item["image"] for _item in json_data[split]]


def prepare_data(
    files: list,
    reports_csv: str,
    cache_rate: float,
    num_workers: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    csv_col_volume: str = "VolumeName",
    csv_col_findings: str = "Findings_EN",
    csv_col_impressions: str = "Impressions_EN",
    expected_shape: tuple = (1, 512, 512, 128),
) -> DataLoader:
    rep = pd.read_csv(resolve_repo_path(reports_csv))
    rep[csv_col_volume] = rep[csv_col_volume].astype(str)

    rep["label_text"] = rep.apply(
        lambda r: _build_label_text(r.get(csv_col_findings, ""), r.get(csv_col_impressions, "")),
        axis=1,
    )

    volume_text_mapping = {
        row[csv_col_volume]: f"Findings: {str(row.get(csv_col_findings,''))} Impression: {str(row.get(csv_col_impressions,''))}"
        for _, row in rep.iterrows()
    }

    # mapping VolumeName -> label_text (per regex)
    volume_labeltext_mapping = {row["VolumeName"]: row["label_text"] for _, row in rep.iterrows()}

    # ---- precompute label mappings (NO tumor_type) ----
    figo_map = {}
    asc_map = {}
    het_map = {}
    shape_maps = {s: {} for s in SHAPE_WORDS}

    for vol, txt in volume_labeltext_mapping.items():
        figo = extract_figo_stage(txt)          # 3/4/None
        asc = extract_ascites(txt)              # 0/1/None
        het = extract_heterogeneity(txt)        # mild/moderate/marked/None
        shapes = extract_shape_flags(txt)       # dict shape->0/1/None

        figo_map[vol] = -1 if figo is None else int(figo)
        asc_map[vol] = -1 if asc is None else int(asc)
        het_map[vol] = -1 if het is None else int(HET_MAP[het])
        for s in SHAPE_WORDS:
            shape_maps[s][vol] = -1 if shapes.get(s, None) is None else int(shapes[s])

    def lookup_text(volume_name: str) -> str:
        return volume_text_mapping.get(str(volume_name), "")

    def lookup_figo(volume_name: str) -> int:
        return figo_map.get(str(volume_name), -1)

    def lookup_ascites(volume_name: str) -> int:
        return asc_map.get(str(volume_name), -1)

    def lookup_heterogeneity(volume_name: str) -> int:
        return het_map.get(str(volume_name), -1)

    def make_lookup_shape(s: str):
        def _f(volume_name: str) -> int:
            return shape_maps[s].get(str(volume_name), -1)
        return _f

    transforms_list = [
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
    ]

    def _check_shape(x, expected_shape=(1, 512, 512, 128)):
        if tuple(x.shape) != tuple(expected_shape):
            raise ValueError(f"Unexpected volume shape {tuple(x.shape)} (expected {expected_shape})")
        return x

    transforms_list.append(
        monai.transforms.Lambdad(
            keys="image",
            func=lambda x: _check_shape(x, expected_shape)
        )
    )

    # riempi testo CLIP usando volume_name come chiave
    transforms_list.append(monai.transforms.Lambdad(keys="impression", func=lookup_text))

    # aggiungi label fields nel batch (da volume_name)

    transforms_list.append(monai.transforms.Lambdad(keys="volume_name", func=str))  # normalizza
    transforms_list.append(monai.transforms.Lambdad(keys="label_figo", func=lookup_figo))
    transforms_list.append(monai.transforms.Lambdad(keys="label_ascites", func=lookup_ascites))
    transforms_list.append(monai.transforms.Lambdad(keys="label_heterogeneity", func=lookup_heterogeneity))
    for s in SHAPE_WORDS:
        transforms_list.append(monai.transforms.Lambdad(keys=f"label_shape_{s}", func=make_lookup_shape(s)))

    tfm = Compose(transforms_list)

    ds = monai.data.CacheDataset(data=files, transform=tfm, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


############## CAMBIO MIO #################
# def build_files_from_json(json_path: str, split: str) -> list:
#     paths = load_filenames(json_path, split=split)
#     out = []
#     for image_path in paths:
#         if not os.path.exists(image_path):
#             continue
#         volume_name = Path(image_path).parent.name  # coerente col tuo training
#         out.append({
#             "image": image_path,
#             "volume_name": volume_name,  # chiave per testo/label da CSV
#             "impression": volume_name,   # placeholder: verrà rimpiazzata col testo completo
#         })
#     return out

def build_files_from_json(json_path: str, split: str) -> list:
    paths = load_filenames(json_path, split=split)
    out = []
    missing_paths = []
    for image_path in paths:
        resolved_path = resolve_volume_path(image_path)
        if resolved_path is None:
            missing_paths.append(image_path)
            continue

        volume_name = resolved_path.parent.name  # coerente col tuo training

        out.append({
            "image": str(resolved_path),
            "volume_name": volume_name,

            # testo: verrà rimpiazzato da lookup_text (Findings+Impression)
            "impression": volume_name,
            
            ####################### FIX MIO #####################
            # --- PRE-CREA LABEL KEYS (necessario per Lambdad) ---
            # "label_figo": -1,
            # "label_ascites": -1,
            # "label_heterogeneity": -1,
            # "label_shape_ovoid": -1,
            # "label_shape_lobulated": -1,
            # "label_shape_elongated": -1,
            # "label_shape_flattened": -1,

            # --- PRE-CREA LABEL KEYS: placeholder = volume_name ---
            # così Lambdad(lookup_*) riceve "IEO...." e può mappare davvero
            "label_figo": volume_name,
            "label_ascites": volume_name,
            "label_heterogeneity": volume_name,
            "label_shape_ovoid": volume_name,
            "label_shape_lobulated": volume_name,
            "label_shape_elongated": volume_name,
            "label_shape_flattened": volume_name,

            ###################### FINE FIX #################

        })
    return out




# ============================================================================
# Metrics (coerenti con Clip_Training_Script.py)
# ============================================================================
@torch.no_grad()
def compute_eval_loss_like_training(config, dataloader, model) -> float:
    model.eval()
    total = 0.0
    n_steps = 0
    device = torch.device(config.device)

    n_gpu = int(getattr(config, "n_gpu", 1))
    for batch in tqdm(dataloader, desc="Loss"):
        images = batch["image"].to(device)
        texts = batch["impression"]

        tokenizer = model.module.tokenizer if (n_gpu > 1 and isinstance(model, torch.nn.DataParallel)) else model.tokenizer
        max_length = model.module.max_length if (n_gpu > 1 and isinstance(model, torch.nn.DataParallel)) else model.max_length

        toks = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(device)

        with torch.amp.autocast("cuda", enabled=getattr(config, "use_amp", False)):
            image_features = model(images, "encode_vision")
            text_features = model(toks, "encode_text")

            if n_gpu > 1 and isinstance(model, torch.nn.DataParallel):
                logit_scale = model.module.model.logit_scale.exp()
            else:
                logit_scale = model.model.logit_scale.exp()

        if image_features.dim() == 3:
            image_features = image_features.mean(dim=1)
        if text_features.dim() == 3:
            text_features = text_features.mean(dim=1)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits_per_image = logit_scale * (image_features @ text_features.t())
        logits_per_text = logits_per_image.t()

        labels = torch.arange(image_features.shape[0], device=logits_per_image.device)
        image_loss = F.cross_entropy(logits_per_image, labels)
        text_loss = F.cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2

        if n_gpu > 1 and isinstance(model, torch.nn.DataParallel):
            loss = loss.mean()

        total += loss.item()
        n_steps += 1

    model.train()
    return total / max(1, n_steps)


@torch.no_grad()
def extract_embeddings(model, dataloader, config):
    model.eval()
    device = torch.device(config.device)
    n_gpu = int(getattr(config, "n_gpu", 1))

    m = model.module if (n_gpu > 1 and isinstance(model, torch.nn.DataParallel)) else model
    tokenizer = m.tokenizer
    max_length = m.max_length

    img_embs, txt_embs = [], []
    labels_accum = {
        "figo": [],
        "ascites": [],
        "heterogeneity": [],
        "shape_ovoid": [],
        "shape_lobulated": [],
        "shape_elongated": [],
        "shape_flattened": [],
    }

    for batch in tqdm(dataloader, desc="Embeddings"):
        images = batch["image"].to(device)
        texts = batch["impression"]

        toks = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].to(device)

        with torch.amp.autocast("cuda", enabled=getattr(config, "use_amp", False)):
            v = model(images, "encode_vision")
            t = model(toks, "encode_text")

            if v.dim() == 3: v = v.mean(dim=1)
            if t.dim() == 3: t = t.mean(dim=1)

            v = v / v.norm(dim=-1, keepdim=True)
            t = t / t.norm(dim=-1, keepdim=True)

        img_embs.append(v.detach().float().cpu())
        txt_embs.append(t.detach().float().cpu())

        # labels (sempre presenti: create dal transform; se manca -> KeyError, quindi ci accorgiamo subito)
        labels_accum["figo"].append(torch.as_tensor(batch["label_figo"]).cpu())
        labels_accum["ascites"].append(torch.as_tensor(batch["label_ascites"]).cpu())
        labels_accum["heterogeneity"].append(torch.as_tensor(batch["label_heterogeneity"]).cpu())
        labels_accum["shape_ovoid"].append(torch.as_tensor(batch["label_shape_ovoid"]).cpu())
        labels_accum["shape_lobulated"].append(torch.as_tensor(batch["label_shape_lobulated"]).cpu())
        labels_accum["shape_elongated"].append(torch.as_tensor(batch["label_shape_elongated"]).cpu())
        labels_accum["shape_flattened"].append(torch.as_tensor(batch["label_shape_flattened"]).cpu())

    V = torch.cat(img_embs, dim=0)
    T = torch.cat(txt_embs, dim=0)
    L = {k: torch.cat(v, dim=0).view(-1) for k, v in labels_accum.items()}

    model.train()
    return V, T, L



def retrieval_metrics_training_style(sim: torch.Tensor) -> Dict[str, float]:
    N = sim.size(0)
    ranks = []
    for i in range(N):
        s = sim[i]
        gt = s[i].item()
        rank = 1 + int((s > gt).sum().item())
        ranks.append(rank)

    ranks_t = torch.tensor(ranks)
    return {
        "R@1": (ranks_t <= 1).float().mean().item(),
        "R@5": (ranks_t <= 5).float().mean().item(),
        "R@10": (ranks_t <= 10).float().mean().item(),
        "median_rank": float(torch.median(ranks_t).item()),
    }


# ============================================================================
# Main
# ============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Valutazione CLIP3D (coerente col training)")
    ap.add_argument("--config", required=True, help="Path config yaml/json usata per CLIP training")
    ap.add_argument("--ckpt", type=str, default=None, help="Checkpoint da valutare (.pt)")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--outdir", type=str, default="eval_clip_metrics")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    logger = get_logger()

    config = load_config_file(args.config)
    mkdir(args.outdir)

    eval_cfg     = getattr(config, "evaluation", {})
    csv_cols     = getattr(eval_cfg, "csv_columns", {})
    lp_cfg       = getattr(eval_cfg, "linear_probing", {})
    exp_shape    = tuple(getattr(eval_cfg, "expected_volume_shape", [1, 512, 512, 128]))

    col_volume      = csv_cols.get("volume_name",   "VolumeName")    if isinstance(csv_cols, dict) else "VolumeName"
    col_findings    = csv_cols.get("findings",      "Findings_EN")   if isinstance(csv_cols, dict) else "Findings_EN"
    col_impressions = csv_cols.get("impressions",   "Impressions_EN") if isinstance(csv_cols, dict) else "Impressions_EN"
    lp_enabled      = lp_cfg.get("enabled", True)   if isinstance(lp_cfg, dict) else True
    lp_max_iter     = lp_cfg.get("max_iter", 2000)  if isinstance(lp_cfg, dict) else 2000
    lp_solver       = lp_cfg.get("solver", "lbfgs") if isinstance(lp_cfg, dict) else "lbfgs"
    # device/n_gpu come training
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count()
    set_seed(seed=getattr(config, "seed", args.seed), n_gpu=config.n_gpu)

    logger.info(f"Device: {config.device} | n_gpu: {config.n_gpu}")

    # ------------------------------------------------------------------
    # 1) Build model esattamente come training
    # ------------------------------------------------------------------
    cfgm = model_cfg_bank()(config.clip_model)
    model = get_model()(cfgm)
    model = model.to(config.device)

    # ------------------------------------------------------------------
    # 2) Load checkpoint (AUTH vs LoRA)
    # ------------------------------------------------------------------
    ckpt_to_load = args.ckpt or getattr(config, "init_ckpt", None)
    if ckpt_to_load:
        ckpt_to_load = str(resolve_repo_path(ckpt_to_load))
        if args.ckpt:
            logger.info(f"Loading ckpt: {ckpt_to_load}")
        else:
            logger.info(f"No --ckpt supplied; loading config.init_ckpt as base checkpoint: {ckpt_to_load}")
        ckpt_obj = torch.load(ckpt_to_load, map_location="cpu", weights_only=False)
        sd = _extract_state_dict(ckpt_obj)
        sd = _strip_module_prefix(sd)

        model = _apply_lora_wrappers_if_needed(config, model, sd, logger)
        debug_lora_devices(model, prefix="[AFTER apply_lora] ")

        missing, unexpected = model.load_state_dict(sd, strict=False)
        debug_lora_devices(model, prefix="[AFTER load_state_dict] ")

        # DEBUG: verifica che i pesi LoRA esistano, non siano tutti zero e siano su CUDA
        lora_params = [(n,p) for n,p in model.named_parameters() if "lora_" in n]
        print("N lora params:", len(lora_params))
        print("First 5:", [(n, p.device, float(p.abs().mean())) for n,p in lora_params[:5]])


        if len(missing) > 0:
            logger.info("Missing example: " + str(missing[:10]))
        if len(unexpected) > 0:
            logger.info("Unexpected example: " + str(unexpected[:10]))

        model = model.to(config.device) ##################### AGGIUNTO DA ME ######################
        debug_lora_devices(model, prefix="[AFTER model.to(device)] ")

        logger.info(f"load_state_dict(strict=False): missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) and len(missing) <= 20:
            logger.info(f"Missing (<=20): {missing}")
        if len(unexpected) and len(unexpected) <= 20:
            logger.info(f"Unexpected (<=20): {unexpected}")
    else:
        logger.warning("Nessun checkpoint fornito e config.init_ckpt assente: valuti un modello random.")

    if config.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # ------------------------------------------------------------------
    # 3) Build splits come training
    # ------------------------------------------------------------------
    train_files = build_files_from_json(config.data_list, split="training")

    val_json = getattr(config, "val_data_list", None)
    if val_json is None:
        raise ValueError("config.val_data_list mancante: nel training usavi un JSON separato per validation.")
    val_files = build_files_from_json(val_json, split="validation")

    test_json = getattr(config, "test_data_list", None)
    test_files = build_files_from_json(test_json, split="test") if test_json else []
    if args.split == "test" and not test_files:
        raise ValueError(
            "SPLIT=test richiede config.test_data_list con almeno un volume. "
            "Evito il fallback su validation per non contaminare la valutazione."
        )

    # split per le metriche principali (loss/retrieval/clip score) basato su args.split
    if args.split == "train":
        eval_files = train_files
    elif args.split == "val":
        eval_files = val_files
    else:
        eval_files = test_files

    logger.info(f"Eval split={args.split} | N={len(eval_files)}")

    # ------------------------------------------------------------------
    # 4) Dataloader per metriche principali
    # ------------------------------------------------------------------
    eval_loader = prepare_data(
        eval_files,
        reports_csv=config.reports_csv,
        cache_rate=0,
        batch_size=1,
        num_workers=int(getattr(config, "num_workers", 2)),
        shuffle=False,
        drop_last=False,
        csv_col_volume=col_volume,
        csv_col_findings=col_findings,
        csv_col_impressions=col_impressions,
        expected_shape=exp_shape,
    )
    
    # safety check: prepare_data should return a DataLoader
    if eval_loader is None:
        logger.error("prepare_data returned None for eval_loader. Controlla data_list/config e i path delle immagini.")
        sys.exit(2)
    

    ############## FIX MIO ####################
    # loss_loader = prepare_data(
    #     eval_files,
    #     reports_csv=config.reports_csv,
    #     cache_rate=0,
    #     batch_size=int(getattr(config, "per_gpu_train_batch_size", 1)),
    #     num_workers=int(getattr(config, "num_workers", 2)),
    #     shuffle=False,
    #     drop_last=False,
    # )

    # --- Loss loader: batch_size deve essere >=2, altrimenti CE su 1x1 = 0 ---
    
    bs_loss = int(getattr(config, "per_gpu_train_batch_size", 4))
    bs_loss = max(2, bs_loss)
    loss_loader = None
    if len(eval_files) < 2:
        logger.warning("Eval split ha meno di 2 campioni: salto la loss training-style.")
    else:
        bs_loss = min(bs_loss, len(eval_files))  # evita batch > N
        loss_loader = prepare_data(
            eval_files,
            reports_csv=config.reports_csv,
            cache_rate=0,
            batch_size=bs_loss,
            num_workers=int(getattr(config, "num_workers", 2)),
            shuffle=False,
            drop_last=False,
            csv_col_volume=col_volume,
            csv_col_findings=col_findings,
            csv_col_impressions=col_impressions,
            expected_shape=exp_shape,
        )

    ################### FINE FIX #####################

    # ------------------------------------------------------------------
    # 5) Dataloader per linear probing (split fisso, NO train_test_split)
    #    train_probing = training + validation
    #    test_probing  = test
    # ------------------------------------------------------------------
    probe_train_loader = None
    probe_test_loader = None
    if lp_enabled:
        if not test_files:
            raise ValueError(
                "linear_probing.enabled=true richiede config.test_data_list. "
                "Disabilitalo con evaluation.linear_probing.enabled=false se vuoi solo loss/retrieval."
            )

        probe_train_files = train_files + val_files
        probe_test_files = test_files

        logger.info(f"Linear probing train = train+val: N={len(probe_train_files)}")
        logger.info(f"Linear probing test  = test:      N={len(probe_test_files)}")

        probe_train_loader = prepare_data(
            probe_train_files,
            reports_csv=config.reports_csv,
            cache_rate=0,
            batch_size=1,
            num_workers=int(getattr(config, "num_workers", 2)),
            shuffle=False,
            drop_last=False,
            csv_col_volume=col_volume,
            csv_col_findings=col_findings,
            csv_col_impressions=col_impressions,
            expected_shape=exp_shape,
        )

        probe_test_loader = prepare_data(
            probe_test_files,
            reports_csv=config.reports_csv,
            cache_rate=0,
            batch_size=1,
            num_workers=int(getattr(config, "num_workers", 2)),
            shuffle=False,
            drop_last=False,
            csv_col_volume=col_volume,
            csv_col_findings=col_findings,
            csv_col_impressions=col_impressions,
            expected_shape=exp_shape,
        )
    else:
        logger.info("Linear probing disabled by evaluation.linear_probing.enabled=false")

    results: Dict[str, Any] = {
        "model_name": getattr(config, "name", "clip3d"),
        "split": args.split,
        "linear_probing_enabled": bool(lp_enabled),
    }

    # ------------------------------------------------------------------
    # 6) Loss
    # ------------------------------------------------------------------
    logger.info("=== Loss (training-style) ===")
    if loss_loader is None:
        results["loss"] = float("nan")
        logger.warning("Skipping training-style loss because the loss loader is unavailable.")
    else:
        loss = compute_eval_loss_like_training(config, loss_loader, model)
        results["loss"] = float(loss)
        logger.info(f"loss = {loss:.6f}")

    # ------------------------------------------------------------------
    # 7) Embeddings + Retrieval + CLIP score (sullo split scelto)
    # ------------------------------------------------------------------
    logger.info("=== Embeddings + Retrieval ===")
    V, T, _ = extract_embeddings(model, eval_loader, config)

    sim_i2t = V @ T.T
    sim_t2i = T @ V.T

    metr_i2t = retrieval_metrics_training_style(sim_i2t)
    metr_t2i = retrieval_metrics_training_style(sim_t2i)

    for k, v in metr_i2t.items():
        results[f"retrieval_i2t_{k}"] = float(v)
    for k, v in metr_t2i.items():
        results[f"retrieval_t2i_{k}"] = float(v)

    clip_diag = torch.diag(sim_i2t).mean().item()
    results["clip_score_diag_mean"] = float(clip_diag)

    logger.info(f"I2T: {metr_i2t} | T2I: {metr_t2i} | diag={clip_diag:.6f}")

    # ------------------------------------------------------------------
    # 8) Linear probing con split fisso (train+val vs test)
    # ------------------------------------------------------------------
    if not lp_enabled:
        logger.info("=== Linear probing skipped ===")
        out_csv = os.path.join(args.outdir, f"clip_eval_{args.split}.csv")
        pd.DataFrame([results]).to_csv(out_csv, index=False)
        logger.info(f"Saved: {out_csv}")
        logger.info(f"Results: {results}")
        return

    logger.info("=== Linear probing (fixed split: train+val vs test) ===")
    Vtr, _, Ltr = extract_embeddings(model, probe_train_loader, config)
    Vte, _, Lte = extract_embeddings(model, probe_test_loader, config)

    X_train = Vtr.numpy()
    X_test  = Vte.numpy()

    def _probe_one(task_name: str, ytr_t: torch.Tensor, yte_t: torch.Tensor):
        y_train = ytr_t.numpy()
        y_test  = yte_t.numpy()

        # tieni solo label valide (>=0)
        tr_mask = y_train >= 0
        te_mask = y_test >= 0

        if tr_mask.sum() < 2 or te_mask.sum() < 2:
            logger.warning(f"[{task_name}] skip: troppo pochi esempi validi (train={tr_mask.sum()} test={te_mask.sum()})")
            return None

        Xtr = X_train[tr_mask]
        Xte = X_test[te_mask]
        ytr = y_train[tr_mask]
        yte = y_test[te_mask]

        if len(np.unique(ytr)) < 2:
            logger.warning(f"[{task_name}] skip: una sola classe in train")
            return None

        if len(np.unique(yte)) < 2:
            logger.warning(f"[{task_name}] skip: una sola classe in test")
            return None

        clf = LogisticRegression(max_iter=lp_max_iter, solver=lp_solver, class_weight="balanced")
        clf.fit(Xtr, ytr)

        ypred = clf.predict(Xte)
        acc = accuracy_score(yte, ypred)

        proba = clf.predict_proba(Xte)
        if proba.shape[1] == 2:
            auc = roc_auc_score(yte, proba[:, 1])
        else:
            auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro")

        return {"n_test": int(len(yte)), "acc": float(acc), "auc": float(auc)}

    # FIGO: tieni solo 3/4
    figo_tr = Ltr["figo"]
    figo_te = Lte["figo"]
    # (qui -1 già escluso dal filtro >=0, ma togliamo anche eventuali altri valori)
    figo_tr = torch.where((figo_tr == 3) | (figo_tr == 4), figo_tr, torch.tensor(-1))
    figo_te = torch.where((figo_te == 3) | (figo_te == 4), figo_te, torch.tensor(-1))

    tasks = {
        "figo_3v4": (figo_tr, figo_te),
        "ascites": (Ltr["ascites"], Lte["ascites"]),
        "heterogeneity": (Ltr["heterogeneity"], Lte["heterogeneity"]),
        "shape_ovoid": (Ltr["shape_ovoid"], Lte["shape_ovoid"]),
        "shape_lobulated": (Ltr["shape_lobulated"], Lte["shape_lobulated"]),
        "shape_elongated": (Ltr["shape_elongated"], Lte["shape_elongated"]),
        "shape_flattened": (Ltr["shape_flattened"], Lte["shape_flattened"]),
    }

    for name, (ytr_t, yte_t) in tasks.items():
        out = _probe_one(name, ytr_t, yte_t)
        if out is None:
            continue
        results[f"probe_{name}_n_test"] = out["n_test"]
        results[f"probe_{name}_acc"] = out["acc"]
        results[f"probe_{name}_auc"] = out["auc"]
        logger.info(f"[PROBE {name}] n_test={out['n_test']} acc={out['acc']:.4f} auc={out['auc']:.4f}")

    # ------------------------------------------------------------------
    # 9) Save CSV
    # ------------------------------------------------------------------
    out_csv = os.path.join(args.outdir, f"clip_eval_{args.split}.csv")
    pd.DataFrame([results]).to_csv(out_csv, index=False)
    logger.info(f"Saved: {out_csv}")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
