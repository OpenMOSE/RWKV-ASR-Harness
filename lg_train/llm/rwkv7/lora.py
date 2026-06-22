# -*- coding: utf-8 -*-
"""LoRA for RWKV7 — independently injectable into time-mix (att) and FFN (cmix).

Targets:
  time-mix : receptance, key, value, output   (the 4 full CxC projections)
  ffn      : key (C->4C), value (4C->C)

The architecture's own low-rank params (att.w1/w2, a1/a2, v1/v2, g1/g2) are
already low-rank and are NOT touched here.

LoRALinear reuses the base Linear's weight tensor (frozen) under the SAME
attribute name `.weight`, so a base checkpoint still loads cleanly; only the
extra `.lora_A` / `.lora_B` (+ `.scaling` buffer) are new. B is zero-init so
ΔW = 0 at start (the model is unchanged until LoRA learns).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        # reuse (and freeze) the base parameters under their original names
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.r = r
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)   # B stays 0 -> dW = 0 at init
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # scaling stored as a buffer so it round-trips through the checkpoint
        self.register_buffer("scaling", torch.tensor(float(alpha) / r))

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        lora = (self.drop(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + lora * self.scaling


def _wrap(module, attr, r, alpha, dropout):
    setattr(module, attr, LoRALinear(getattr(module, attr), r, alpha, dropout))


def inject_lora(llm, tmix_rank=0, ffn_rank=0, alpha=0.0, dropout=0.0):
    """Wrap target Linears in `llm` (an RWKV7) with LoRA. Returns #wrapped layers.

    alpha<=0 -> alpha = rank (scaling = 1.0). tmix and ffn can use different ranks.
    """
    n = 0
    for blk in llm.blocks:
        if tmix_rank > 0:
            a = alpha if alpha > 0 else tmix_rank
            for attr in ("receptance", "key", "value", "output"):
                _wrap(blk.att, attr, tmix_rank, a, dropout)
                n += 1
        if ffn_rank > 0:
            a = alpha if alpha > 0 else ffn_rank
            for attr in ("key", "value"):
                _wrap(blk.ffn, attr, ffn_rank, a, dropout)
                n += 1
    return n


def detect_lora_ranks(state_dict):
    """Inspect a checkpoint and return (tmix_rank, ffn_rank) from lora_A shapes."""
    tmix_rank = ffn_rank = 0
    for k, v in state_dict.items():
        if k.endswith(".lora_A"):
            r = v.shape[0]
            if ".att." in k:
                tmix_rank = r
            elif ".ffn." in k:
                ffn_rank = r
    return tmix_rank, ffn_rank
