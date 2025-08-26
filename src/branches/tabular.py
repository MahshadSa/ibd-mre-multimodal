import torch, torch.nn as nn

class TabularNormalizer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std",  torch.ones(n_features))
    def fit(self, X: torch.Tensor):
        self.mean = X.mean(dim=0)
        self.std  = X.std(dim=0).clamp(min=1e-6)
    def forward(self, X: torch.Tensor):
        return (X - self.mean) / self.std

class TabularBranch(nn.Module):
    def __init__(self, in_dim=10, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)
