"""SSL-AASIST -- self-supervised front-end + AASIST graph back-end.

Faithful port of TakHemlata/SSL_Anti-spoofing `model.py` (the model RADAR
used as its baseline), with two substitutions:
  * fairseq XLS-R  ->  HuggingFace Wav2Vec2Model (wav2vec2-base by default;
    facebook/wav2vec2-xls-r-300m for the cross-lingual target)
  * the AASIST graph components are the vendored clovaai/aasist ones.

Backbone frozen by default ("frozen backbone first", rebuild plan Sec 4.5):
train only LL + encoder + graph. Partial fine-tuning is a later ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from voiceguard.models._vendor.aasist_ref import (
    GraphAttentionLayer,
    GraphPool,
    HtrgGraphAttentionLayer,
)

DEFAULT_SSL = "facebook/wav2vec2-base"


class _ResBlock(nn.Module):
    """The SSL-AASIST encoder block (TakHemlata/SSL_Anti-spoofing): conv-conv
    residual, **no max-pool** -- SSL feature maps are already short in time,
    so the clovaai block's MaxPool2d would collapse them to zero length.
    """

    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], (2, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], (2, 3), padding=(0, 1))
        self.selu = nn.SELU(inplace=True)
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_ds = nn.Conv2d(nb_filts[0], nb_filts[1], (1, 3), padding=(0, 1))

    def forward(self, x):
        identity = x
        out = x if self.first else self.selu(self.bn1(x))
        out = self.conv1(out)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_ds(identity)
        return out + identity


class SSLFrontend(nn.Module):
    """HF wav2vec2 / XLS-R feature extractor. forward(wav) -> (B, T, D)."""

    def __init__(self, name: str = DEFAULT_SSL, freeze: bool = True, weighted_sum: bool = True):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.model = Wav2Vec2Model.from_pretrained(name)
        self.out_dim = int(self.model.config.hidden_size)
        self.freeze = freeze
        self.weighted_sum = weighted_sum
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()
        if weighted_sum:
            n = self.model.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.zeros(n))

    def train(self, mode: bool = True):  # keep a frozen backbone in eval
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav[..., 0]
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.model(wav, output_hidden_states=self.weighted_sum)
        if self.weighted_sum:
            hs = torch.stack(out.hidden_states, dim=0)  # (L, B, T, D)
            w = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            return (w * hs).sum(dim=0)
        return out.last_hidden_state


class SSLAASIST(nn.Module):
    def __init__(self, ssl_name: str = DEFAULT_SSL, freeze_ssl: bool = True):
        super().__init__()
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temps = [2.0, 2.0, 100.0, 100.0]

        self.ssl = SSLFrontend(ssl_name, freeze=freeze_ssl, weighted_sum=True)
        self.LL = nn.Linear(self.ssl.out_dim, 128)

        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        self.encoder = nn.Sequential(
            _ResBlock(filts[1], first=True),
            _ResBlock(filts[2]),
            _ResBlock(filts[3]),
            _ResBlock(filts[4]),
            _ResBlock(filts[4]),
            _ResBlock(filts[4]),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, (1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, (1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))  # 128 // pool(3) = 42
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temps[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temps[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temps[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temps[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temps[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temps[2]
        )
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.ssl(x)  # (B, T, D)
        x = self.LL(feat).transpose(1, 2).unsqueeze(1)  # (B, 1, 128, T)
        x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))
        x = self.encoder(x)
        x = self.selu(self.first_bn1(x))

        w = self.attention(x)
        m = torch.sum(x * F.softmax(w, dim=-1), dim=-1)
        e_S = m.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        m1 = torch.sum(x * F.softmax(w, dim=-2), dim=-2)
        e_T = m1.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1, out_T1 = self.pool_hS1(out_S1), self.pool_hT1(out_T1)
        oT, oS, mA = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1, out_S1, master1 = out_T1 + oT, out_S1 + oS, master1 + mA

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2, out_T2 = self.pool_hS2(out_S2), self.pool_hT2(out_T2)
        oT, oS, mA = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2, out_S2, master2 = out_T2 + oT, out_S2 + oS, master2 + mA

        out_T = torch.max(self.drop_way(out_T1), self.drop_way(out_T2))
        out_S = torch.max(self.drop_way(out_S1), self.drop_way(out_S2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))

        t_max, _ = torch.max(torch.abs(out_T), dim=1)
        s_max, _ = torch.max(torch.abs(out_S), dim=1)
        last_hidden = torch.cat(
            [t_max, out_T.mean(1), s_max, out_S.mean(1), master.squeeze(1)], dim=1
        )
        # upstream convention: column 1 = bonafide. NeuralCM flips it to column 0.
        return self.out_layer(self.drop(last_hidden))

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)
