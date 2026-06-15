# lora_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    Implementazione LoRA per layer lineari.
    Decompone ΔW = B*A dove A ∈ R^(r×d_in), B ∈ R^(d_out×r)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        
        # Inizializzazione Kaiming per A, zero per B (come in paper LoRA)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B rimane a zero -> ΔW iniziale = 0
        
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [..., in_features]
        Returns:
            LoRA contribution: [..., out_features]
        """
        # x @ A^T -> [..., r]
        # result @ B^T -> [..., out_features]
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        self.linear = linear
        # Aggiungi LoRA
        self.lora = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(
    model: nn.Module,
    target_modules: list[str],
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    verbose: bool = True,
) -> nn.Module:
    """
    Applica LoRA ai moduli specificati del modello.
    """
    # 1. FREEZE GLOBALE
    for param in model.parameters():
        param.requires_grad = False
    lora_applied_count = 0
    # 2. APPLICA LoRA
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                *parent_names, attr_name = name.split('.')
                parent = model
                for pname in parent_names:
                    parent = getattr(parent, pname)
                setattr(
                    parent,
                    attr_name,
                    LinearWithLoRA(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                )
                lora_applied_count += 1
                if verbose:
                    print(f"✓ LoRA applicato a: {name} (in={module.in_features}, out={module.out_features})")
    # 3. FREEZE ESPLICITO LINEAR ORIGINALI
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            for param in module.linear.parameters():
                param.requires_grad = False
    # 4. SBLOCCA PROJECTION
    for name, param in model.named_parameters():
        if 'text_projection' in name:
            param.requires_grad = True
            if verbose:
                print(f"✓ Unfrozen projection: {name}")
    # STATISTICHE
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== STATISTICHE LoRA ===")
        print(f"Layer LoRA applicati: {lora_applied_count}")
        print(f"Parametri totali: {total_params:,}")
        print(f"Parametri trainabili: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    return model


def get_lora_parameters(model: nn.Module) -> list:
    """Estrae tutti i parametri effettivamente trainabili per l'optimizer."""
    return [param for param in model.parameters() if param.requires_grad]
