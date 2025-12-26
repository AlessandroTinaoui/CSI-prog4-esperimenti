# mlp/common/model.py

from typing import List
import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))  # output regressor
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)
