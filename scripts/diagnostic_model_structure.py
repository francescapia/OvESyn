# diagnostic_model_structure.py
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.cfg_helper import model_cfg_bank
from core.models.common.get_model import get_model
import torch

# Carica il modello
cfgm = model_cfg_bank()("clip_3D")
clip = get_model()(cfgm)

# Stampa struttura completa
print("\n=== STRUTTURA MODELLO CLIP3D ===\n")
for name, module in clip.named_modules():
    print(f"{name}: {type(module).__name__}")
    
print("\n=== PARAMETRI TRAINABILI ===\n")
for name, param in clip.named_parameters():
    print(f"{name}: {param.shape}")