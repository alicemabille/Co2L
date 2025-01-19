import torch
import torch.nn as nn

class Trashcan(nn.Module):

    def __init__(self, feat_dim=128):
        self.features = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)