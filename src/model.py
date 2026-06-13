"""Cross-modal attention fusion for plantar-pressure / 2D-pose pairing.

Replaces the original render-to-image + frozen-VGG16 + LSTM pipeline with a
lightweight model that learns directly on the raw sensor signals:

    insole pressure stream  (T, 8)   ->  encoder ->  cross-attention  -> pairing
    skeleton joint  stream  (T, J)   ->  encoder ->  cross-attention  -> logits

The task is *cross-modal pairing*: decide whether an insole sequence and a
skeleton sequence belong to the same person (PAIRED=0 / UNPAIRED=1).
"""

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Per-frame projection + Transformer encoder over the time axis."""

    def __init__(self, in_dim, d_model, nhead, nlayers, dropout, max_len=64):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)

    def forward(self, x):  # x: (B, T, in_dim)
        t = x.size(1)
        h = self.proj(x) + self.pos[:, :t]
        return self.encoder(h)  # (B, T, d_model)


class CrossAttentionBlock(nn.Module):
    """One stream attends to the other; residual + norm."""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, context):
        out, _ = self.attn(query, context, context)
        return self.norm(query + out)


class CrossModalAttentionFusion(nn.Module):
    def __init__(self, insole_dim=8, skel_dim=12, d_model=64, nhead=4,
                 nlayers=2, num_classes=2, dropout=0.1):
        super().__init__()
        self.insole_enc = ModalityEncoder(insole_dim, d_model, nhead, nlayers, dropout)
        self.skel_enc = ModalityEncoder(skel_dim, d_model, nhead, nlayers, dropout)
        self.ins2skel = CrossAttentionBlock(d_model, nhead, dropout)
        self.skel2ins = CrossAttentionBlock(d_model, nhead, dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, insole, skeleton):  # (B, T, 8), (B, T, J)
        hi = self.insole_enc(insole)
        hs = self.skel_enc(skeleton)
        # each modality is contextualised by the other
        ci = self.ins2skel(hi, hs)   # insole queries attend to skeleton
        cs = self.skel2ins(hs, hi)   # skeleton queries attend to insole
        pooled = torch.cat([ci.mean(dim=1), cs.mean(dim=1)], dim=-1)
        return self.head(pooled)
