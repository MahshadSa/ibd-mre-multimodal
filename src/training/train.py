import argparse, os, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics import MetricCollection

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class SyntheticIBDDataset(Dataset):
    """Tri-modal synthetic dataset: (image (1,64,64), tabular (10,), text (128,), label)."""
    def __init__(self, n_samples: int = 400, image_shape=(1,64,64), n_tabular: int = 10, n_text: int = 128, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.img = rng.normal(size=(n_samples, *image_shape)).astype(np.float32)
        self.tab = rng.normal(size=(n_samples, n_tabular)).astype(np.float32)
        self.txt = rng.normal(size=(n_samples, n_text)).astype(np.float32)
        w_img = rng.normal(size=(np.prod(image_shape),))
        w_tab = rng.normal(size=(n_tabular,))
        w_txt = rng.normal(size=(n_text,))
        logits = (self.img.reshape(n_samples, -1) @ w_img + self.tab @ w_tab + self.txt @ w_txt) / 50.0
        probs = 1 / (1 + np.exp(-logits))
        self.y = (probs > 0.5).astype(np.float32)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.img[idx]),
            torch.from_numpy(self.tab[idx]),
            torch.from_numpy(self.txt[idx]),
            torch.tensor(self.y[idx]),
        )

class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*16*16, out_dim), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int = 10, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,out_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class TextEncoder(nn.Module):
    def __init__(self, in_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,out_dim), nn.ReLU())
    def forward(self, x): return self.net(x)

class FusionHead(nn.Module):
    def __init__(self, dims, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sum(dims), hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, feats): return self.net(torch.cat(feats, dim=1))

class Model(nn.Module):
    def __init__(self, use_img=True, use_tab=True, use_txt=True):
        super().__init__()
        self.use_img, self.use_tab, self.use_txt = use_img, use_tab, use_txt
        if use_img: self.img_enc = ImageEncoder(out_dim=128)
        if use_tab: self.tab_enc = TabularEncoder(in_dim=10, out_dim=32)
        if use_txt: self.txt_enc = TextEncoder(in_dim=128, out_dim=64)
        dims = ([] + ([128] if use_img else []) + ([32] if use_tab else []) + ([64] if use_txt else []))
        self.head = FusionHead(dims, hidden=128)

    def forward(self, img, tab, txt):
        feats = []
        if self.use_img: feats.append(self.img_enc(img))
        if self.use_tab: feats.append(self.tab_enc(tab))
        if self.use_txt: feats.append(self.txt_enc(txt))
        logit = self.head(feats).squeeze(1)
        return logit

def train_one_epoch(model, dl, optimizer, loss_fn, device):
    model.train(); total_loss = 0.0
    for img, tab, txt, y in dl:
        img, tab, txt, y = img.to(device), tab.to(device), txt.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(img, tab, txt)
        loss = loss_fn(logits, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    metrics = MetricCollection({"auroc": BinaryAUROC(), "auprc": BinaryAveragePrecision()}).to(device)
    total_loss = 0.0; loss_fn = nn.BCEWithLogitsLoss()
    for img, tab, txt, y in dl:
        img, tab, txt, y = img.to(device), tab.to(device), txt.to(device), y.to(device)
        logits = model(img, tab, txt); loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = torch.sigmoid(logits); metrics.update(preds, y.int())
    m = metrics.compute(); return total_loss / len(dl.dataset), {k: v.item() for k,v in m.items()}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-only", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--tabular-only", action="store_true")
    p.add_argument("--save-dir", type=str, default="artifacts")
    args = p.parse_args()

    set_seed(args.seed)
    use_img, use_tab, use_txt = True, True, True
    if args.image_only: use_tab = use_txt = False
    if args.text_only:  use_img = use_tab = False
    if args.tabular_only: use_img = use_txt = False

    ds = SyntheticIBDDataset(n_samples=args.samples, seed=args.seed)
    n_train = int(0.8 * len(ds)); n_val = len(ds) - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size)

    device = torch.device(args.device)
    model = Model(use_img=use_img, use_tab=use_tab, use_txt=use_txt).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    os.makedirs(args.save_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_dl, optimizer, loss_fn, device)
        vl, m = evaluate(model, val_dl, device)
        print(f"Epoch {ep:02d} | train_loss={tr:.4f} val_loss={vl:.4f} AUROC={m['auroc']:.3f} AUPRC={m['auprc']:.3f}")

    path = os.path.join(args.save_dir, "mvp_fusion.pt")
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")

if __name__ == "__main__":
    main()
