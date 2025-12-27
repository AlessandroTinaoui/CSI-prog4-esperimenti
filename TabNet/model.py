# model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostBatchNorm1d(nn.Module):
    """
    Ghost BatchNorm: spezza il batch in "virtual batches" per stabilizzare training
    su dataset piccoli/non-IID (molto utile in Federated).
    """
    def __init__(self, num_features: int, virtual_batch_size: int = 128, momentum: float = 0.01):
        super().__init__()
        self.num_features = num_features
        self.virtual_batch_size = max(1, int(virtual_batch_size))
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or x.size(0) <= self.virtual_batch_size:
            return self.bn(x)

        chunks = x.chunk((x.size(0) + self.virtual_batch_size - 1) // self.virtual_batch_size, dim=0)
        out = [self.bn(c) for c in chunks]
        return torch.cat(out, dim=0)


class GLULayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bn_virtual_bs: int, bn_momentum: float):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2, bias=False)
        self.bn = GhostBatchNorm1d(out_dim * 2, virtual_batch_size=bn_virtual_bs, momentum=bn_momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(b)


class FeatureTransformer(nn.Module):
    """
    Feature transformer: shared blocks + independent blocks, con residual scaling sqrt(0.5)
    """
    def __init__(
        self,
        input_dim: int,
        n_d: int,
        n_a: int,
        n_shared: int,
        n_independent: int,
        bn_virtual_bs: int,
        bn_momentum: float,
    ):
        super().__init__()
        out_dim = n_d + n_a

        self.shared: nn.ModuleList = nn.ModuleList()
        prev = input_dim
        for _ in range(n_shared):
            self.shared.append(GLULayer(prev, out_dim, bn_virtual_bs, bn_momentum))
            prev = out_dim

        self.independent: nn.ModuleList = nn.ModuleList()
        prev = out_dim if n_shared > 0 else input_dim
        for i in range(n_independent):
            self.independent.append(GLULayer(prev, out_dim, bn_virtual_bs, bn_momentum))
            prev = out_dim

        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.shared) > 0:
            for layer in self.shared:
                h = layer(x)
                x = (h + x) * 0.70710678 if h.shape == x.shape else h
        if len(self.independent) > 0:
            for layer in self.independent:
                h = layer(x)
                x = (h + x) * 0.70710678
        return x


class AttentiveTransformer(nn.Module):
    def __init__(self, in_dim: int, input_dim: int, bn_virtual_bs: int, bn_momentum: float):
        super().__init__()
        self.fc = nn.Linear(in_dim, input_dim, bias=False)
        self.bn = GhostBatchNorm1d(input_dim, virtual_batch_size=bn_virtual_bs, momentum=bn_momentum)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.bn(x)
        x = x * prior
        # Sparsemax sarebbe più fedele al paper; qui uso entmax-like softmax stabile.
        return F.softmax(x, dim=-1)


@dataclass
class TabNetConfig:
    n_d: int = 16
    n_a: int = 16
    n_steps: int = 4
    gamma: float = 1.3
    n_shared: int = 2
    n_independent: int = 2
    bn_virtual_bs: int = 128
    bn_momentum: float = 0.02


class TabNetRegressor(nn.Module):
    """
    TabNet per regressione su tabular data.
    Output: y_norm (float) -> de-normalizzato fuori dal modello.
    """
    def __init__(self, input_dim: int, cfg: Optional[TabNetConfig] = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.cfg = cfg or TabNetConfig()

        n_d, n_a = self.cfg.n_d, self.cfg.n_a
        n_steps = self.cfg.n_steps

        self.initial_bn = GhostBatchNorm1d(self.input_dim, virtual_batch_size=self.cfg.bn_virtual_bs, momentum=self.cfg.bn_momentum)

        # shared transformer: usato in ogni step (come nel paper)
        self.shared_transformer = FeatureTransformer(
            input_dim=self.input_dim,
            n_d=n_d,
            n_a=n_a,
            n_shared=self.cfg.n_shared,
            n_independent=0,
            bn_virtual_bs=self.cfg.bn_virtual_bs,
            bn_momentum=self.cfg.bn_momentum,
        )

        # per-step transformers e attentive transformers
        self.step_transformers = nn.ModuleList()
        self.attentive = nn.ModuleList()

        for _ in range(n_steps):
            self.step_transformers.append(
                FeatureTransformer(
                    input_dim=n_d + n_a,
                    n_d=n_d,
                    n_a=n_a,
                    n_shared=0,
                    n_independent=self.cfg.n_independent,
                    bn_virtual_bs=self.cfg.bn_virtual_bs,
                    bn_momentum=self.cfg.bn_momentum,
                )
            )
            self.attentive.append(
                AttentiveTransformer(
                    in_dim=n_a,
                    input_dim=self.input_dim,
                    bn_virtual_bs=self.cfg.bn_virtual_bs,
                    bn_momentum=self.cfg.bn_momentum,
                )
            )

        self.fc_out = nn.Linear(n_d, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim]
        x = self.initial_bn(x)

        prior = torch.ones(x.size(0), self.input_dim, device=x.device, dtype=x.dtype)
        masked_x = x

        out_agg = 0.0

        # initial shared block
        h = self.shared_transformer(masked_x)  # [B, n_d+n_a]

        for step in range(self.cfg.n_steps):
            h_step = self.step_transformers[step](h)  # [B, n_d+n_a]
            d = h_step[:, : self.cfg.n_d]
            a = h_step[:, self.cfg.n_d :]

            # decision output accumulation
            out_agg = out_agg + F.relu(d)

            # compute mask for next step (except last)
            if step < self.cfg.n_steps - 1:
                mask = self.attentive[step](a, prior)  # [B, input_dim]
                # update prior
                prior = prior * (self.cfg.gamma - mask)
                masked_x = mask * x
                h = self.shared_transformer(masked_x)

        y = self.fc_out(out_agg).squeeze(-1)
        return y
