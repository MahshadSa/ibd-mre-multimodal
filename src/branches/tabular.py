import torch
import torch.nn as nn

class TabularNormalizer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std",  torch.ones(n_features))

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        # keep them as buffers
        self.mean.copy_(X.mean(dim=0))
        self.std.copy_(X.std(dim=0, unbiased=False).clamp(min=1e-6))

    def forward(self, X: torch.Tensor):
        return (X - self.mean) / self.std


class TabularMixed(nn.Module):
   
    def __init__(self, cat_cardinalities, n_cont: int, out_dim: int = 32,
                 hidden_dims=(128, 64), dropout=0.1, use_batchnorm=True):
        super().__init__()

        self.embs = nn.ModuleList()
        self.cat_dims = []
        for K in cat_cardinalities:               # K = known categories from TRAIN
            K_eff = K + 1                         # +1 = OOV at index K
            d = min(50, int(round(1.6 * (K_eff ** 0.56))))  
            self.cat_dims.append(d)
            self.embs.append(nn.Embedding(K_eff, d))

        self.n_cont = n_cont
        self.normalizer = TabularNormalizer(n_cont) if n_cont > 0 else None

        in_dim = (sum(self.cat_dims) if self.cat_dims else 0) + (n_cont if n_cont > 0 else 0)
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm: layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.mlp = nn.Sequential(*layers)

    @torch.no_grad()
    def fit_normalizer(self, X_cont: torch.Tensor):
        if self.normalizer is not None and X_cont is not None and X_cont.numel() > 0:
            self.normalizer.fit(X_cont)

    def forward(self, X_cat: torch.Tensor | None, X_cont: torch.Tensor | None):

        parts = []
        if X_cat is not None and len(self.embs) > 0:
            # expect X_cat[:, i] to be in [0..K] (K is OOV id)
            cat_vecs = [emb(X_cat[:, i].long()) for i, emb in enumerate(self.embs)]
            parts.append(torch.cat(cat_vecs, dim=1))
            
        if self.n_cont > 0 and X_cont is not None:
            cont = X_cont.float()
            cont = self.normalizer(cont) if self.normalizer is not None else cont
            parts.append(cont)

        x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        return self.mlp(x)
