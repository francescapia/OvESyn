import json
from pathlib import Path
import pandas as pd
import torch
import monai
from monai.data import DataLoader
from monai.transforms import Compose

import sys
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from clip_training.utils.util import load_config_file
from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model

def load_val_paths(val_json_path: str):
    j = json.load(open(val_json_path))
    # prova chiavi comuni
    for k in ["validation", "val", "testing", "test"]:
        if k in j:
            return [x["image"] for x in j[k]]
    # fallback: se per qualche motivo usa "training"
    if "training" in j:
        return [x["image"] for x in j["training"]]
    raise ValueError(f"Keys in {val_json_path}: {list(j.keys())}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="clip_train_config.yaml usato nel training")
    ap.add_argument("--ckpt", required=True, help="checkpoint .pt del CLIP finetunato")
    ap.add_argument("--val_json", required=True, help="json con lista volumi validation (108)")
    ap.add_argument("--reports_csv", required=True, help="reports.csv (Promptgen vers 1)")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config_file(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # modello
    cfgm = model_cfg_bank()(cfg.clip_model)
    model = get_model()(cfgm).to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # mapping testo
    rep = pd.read_csv(args.reports_csv)
    rep["VolumeName"] = rep["VolumeName"].astype(str).str.strip()
    mapping = {
        r["VolumeName"]: f"Findings: {r['Findings_EN']} Impression: {r['Impressions_EN']}"
        for _, r in rep.iterrows()
    }

    # lista val
    val_paths = load_val_paths(args.val_json)
    items = []
    for p in val_paths:
        pid = Path(p).parent.name
        if pid not in mapping:
            continue
        items.append({"image": p, "pid": pid})

    print("VAL items used:", len(items), " / raw:", len(val_paths))

    # transforms (come training)
    tfm = Compose([
        monai.transforms.LoadImaged(keys=["image"]),
        monai.transforms.EnsureChannelFirstd(keys=["image"]),
        monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
        monai.transforms.ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        monai.transforms.Resized(keys=["image"], spatial_size=(512, 512, 128), mode="trilinear"),
    ])

    ds = monai.data.Dataset(data=items, transform=tfm)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.num_workers, shuffle=False)

    # embeddings
    all_img = []
    all_txt = []
    all_pid = []

    tokenizer = model.tokenizer
    max_length = model.max_length

    with torch.no_grad():
        for batch in dl:
            x = batch["image"].to(device)
            pid_list = batch["pid"]
            texts = [mapping[pid] for pid in pid_list]

            tok = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )["input_ids"].to(device)

            img_f = model(x, "encode_vision")
            txt_f = model(tok, "encode_text")

            # pooling se (B, T, D)
            if img_f.dim() == 3: img_f = img_f.mean(dim=1)
            if txt_f.dim() == 3: txt_f = txt_f.mean(dim=1)

            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

            all_img.append(img_f.cpu())
            all_txt.append(txt_f.cpu())
            all_pid.extend(pid_list)

    img_emb = torch.cat(all_img, dim=0)   # (N,D)
    txt_emb = torch.cat(all_txt, dim=0)   # (N,D)
    N = img_emb.shape[0]

    sim = img_emb @ txt_emb.T  # (N,N), cosine

    def recall_at_k(sim_mat, k):
        # per ogni riga i, correct index = i
        topk = torch.topk(sim_mat, k=k, dim=1).indices
        correct = torch.arange(sim_mat.shape[0]).unsqueeze(1)
        return (topk == correct).any(dim=1).float().mean().item()

    def median_rank(sim_mat):
        # rank della diagonale (1=best)
        ranks = []
        for i in range(sim_mat.shape[0]):
            order = torch.argsort(sim_mat[i], descending=True)
            r = (order == i).nonzero(as_tuple=False).item() + 1
            ranks.append(r)
        ranks = torch.tensor(ranks)
        return int(torch.median(ranks).item())

    print("\nImage -> Text")
    for k in [1,5,10]:
        print(f"R@{k}: {recall_at_k(sim, k):.3f}")
    print("Median rank:", median_rank(sim))

    print("\nText -> Image")
    sim_t = sim.T
    for k in [1,5,10]:
        print(f"R@{k}: {recall_at_k(sim_t, k):.3f}")
    print("Median rank:", median_rank(sim_t))

if __name__ == "__main__":
    main()
