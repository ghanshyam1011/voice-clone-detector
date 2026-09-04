"""Neural countermeasures: thin adapters over the vendored reference
implementations (AASIST, RawNet2).

Two conventions are reconciled here:
  * upstream returns ``(hidden, logits)`` and uses class 1 = bonafide;
  * this repo uses class 0 = bonafide everywhere (baseline, eval harness).
The wrapper drops the hidden output and swaps the logit columns, so
``forward`` returns ``(B, 2)`` logits with **column 0 = bonafide**.

All three models consume a fixed 64600-sample (~4.04 s @ 16 kHz) mono
waveform, matching the reference configs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from voiceguard.models._vendor import aasist_ref, rawnet2_ref

NB_SAMP = 64600
_VENDOR = Path(__file__).parent / "_vendor"

_RAWNET2_CFG = {
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "first_conv": 1024,
    "in_channels": 1,
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 2,
    "nb_samp": NB_SAMP,
}


def _load_conf(name: str) -> dict:
    return json.loads((_VENDOR / name).read_text())["model_config"]


class NeuralCM(nn.Module):
    """Wraps a vendored model. forward(x) -> (B, 2) logits, col 0 = bonafide."""

    def __init__(self, backbone: nn.Module, name: str):
        super().__init__()
        self.backbone = backbone
        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.backbone(x)
        logits = out[1] if isinstance(out, tuple) else out
        return logits.flip(-1)  # [spoof, bona] -> [bona, spoof]

    @torch.no_grad()
    def score_bonafide(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)[:, 0]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_cm(name: str) -> NeuralCM:
    name = name.lower()
    # both reference Model.__init__ mutate their config dict in place -> deep-copy
    if name == "rawnet2":
        backbone = rawnet2_ref.Model(copy.deepcopy(_RAWNET2_CFG))
    elif name == "aasist":
        backbone = aasist_ref.Model(_load_conf("AASIST.conf"))
    elif name in ("aasist-l", "aasist_l"):
        backbone = aasist_ref.Model(_load_conf("AASIST-L.conf"))
    else:
        raise ValueError(f"unknown CM {name!r}; have: rawnet2, aasist, aasist-l")
    return NeuralCM(backbone, name)
