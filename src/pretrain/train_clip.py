import argparse, os, math, torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import models, transforms
from transformers import AutoModel, AutoTokenizer
from src.datasets.paired_csv import PairedImageTextCSV

import numpy as np, random
from PIL import Image

# --------- helpers ---------

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 0):
        super().__init__()
        self.net = (
            nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
            if hidden > 0 else
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x): return nn.functional.normalize(self.net(x), dim=-1)

def cosine_lr(optimizer, base_lr, steps, warmup=0.05):
    warm_steps = int(steps * warmup)
    def schedule(step):
        if step < warm_steps: return (step + 1) / max(1, warm_steps)
        t = (step - warm_steps) / max(1, steps - warm_steps)
        return 0.5 * (1 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

def build_transforms(img_size=224):
    from torchvision import transforms as T
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

# ---- top-level DummyPairs so DataLoader can pickle on Windows ----
class DummyPairs(Dataset):
    """Synthetic image–text pairs for smoke testing."""
    def __init__(self, n=1000, transform=None):
        self.n = n
        self.transform = transform
        self.phrases = [
            "no acute process", "bowel wall thickening", "diffusion restriction",
            "active inflammation", "normal study", "mural hyperenhancement"
        ]
    def __len__(self): return self.n
    def __getitem__(self, i):
        img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype("uint8"))
        if self.transform: img = self.transform(img)
        txt = random.choice(self.phrases)
        return img, txt

# --------- model ---------

class CLIPLike(nn.Module):
    def __init__(self, text_model="distilbert-base-uncased", img_out=512, txt_out=768, proj_dim=256):
        super().__init__()
        rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.image_backbone = nn.Sequential(*list(rn.children())[:-1])  # (B,512,1,1)
        self.img_proj = ProjectionHead(img_out, proj_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_backbone = AutoModel.from_pretrained(text_model)
        hidden = getattr(self.text_backbone.config, "hidden_size", txt_out)
        self.txt_proj = ProjectionHead(hidden, proj_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/0.07)))

    def encode_image(self, x): return self.img_proj(self.image_backbone(x).flatten(1))
    def encode_text(self, ids, attn):
        out = self.text_backbone(input_ids=ids, attention_mask=attn)
        cls = out.last_hidden_state[:, 0]
        return self.txt_proj(cls)

    def forward(self, images, input_ids, attention_mask):
        zi = self.encode_image(images)
        zt = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * zi @ zt.t()
        return logits_per_image, logits_per_image.t()

@torch.no_grad()
def retrieval_eval(model, dl, device, ks=(1,5,10)):
    model.eval()
    img_embs, txt_embs = [], []
    for imgs, texts in dl:
        toks = model.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
        imgs = imgs.to(device)
        toks = {k: v.to(device) for k, v in toks.items()}
        img_embs.append(model.encode_image(imgs))
        txt_embs.append(model.encode_text(toks["input_ids"], toks["attention_mask"]))
    img_embs = torch.cat(img_embs); txt_embs = torch.cat(txt_embs)
    sims = img_embs @ txt_embs.t()
    ranks_it = sims.argsort(dim=1, descending=True)
    ranks_ti = sims.t().argsort(dim=1, descending=True)

    def recall_at(ranks):
        N = ranks.size(0); out = {}
        gold = torch.arange(N, device=ranks.device).unsqueeze(1)
        for k in ks:
            hits = (ranks[:, :k] == gold).any(dim=1).float().mean().item()
            out[f"R@{k}"] = hits
        return out

    r1 = recall_at(ranks_it)
    r2 = {f"T2I_{k}": v for k, v in recall_at(ranks_ti).items()}
    return {**r1, **r2}

# --------- train entrypoint ---------

def main():
    default_workers = 0 if os.name == "nt" else 2

    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default=None, help='CSV with columns image_path,text')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--proj-dim', type=int, default=256)
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--text-model', type=str, default='distilbert-base-uncased')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save-dir', type=str, default='artifacts/clip_pretrain')
    ap.add_argument('--val-split', type=float, default=0.05)
    ap.add_argument('--num-workers', type=int, default=default_workers)
    args = ap.parse_args()

    device = torch.device(args.device)
    model = CLIPLike(text_model=args.text_model, proj_dim=args.proj_dim).to(device)
    tfm = build_transforms(args.img_size)

    if args.csv is None:
        ds = DummyPairs(n=1000, transform=tfm)
    else:
        ds = PairedImageTextCSV(args.csv, transform=tfm)

    n_val = max(1, int(len(ds) * args.val_split)); n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=pin)

    params = [
        {"params": model.image_backbone.parameters(), "lr": args.lr},
        {"params": model.text_backbone.parameters(),  "lr": args.lr},
        {"params": list(model.img_proj.parameters()) + list(model.txt_proj.parameters()) + [model.logit_scale], "lr": args.lr},
    ]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_dl))
    sched = cosine_lr(optim, args.lr, steps=total_steps)

    ce = nn.CrossEntropyLoss()
    for ep in range(1, args.epochs + 1):
        model.train(); running = 0.0
        for imgs, texts in train_dl:
            toks = model.tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors='pt')
            if device.type == "cuda":
                imgs = imgs.to(device, non_blocking=True)
                toks = {k: v.to(device, non_blocking=True) for k, v in toks.items()}
            else:
                imgs = imgs.to(device); toks = {k: v.to(device) for k, v in toks.items()}

            logits_i, logits_t = model(imgs, toks['input_ids'], toks['attention_mask'])
            targets = torch.arange(logits_i.size(0), device=device)
            loss = (ce(logits_i, targets) + ce(logits_t, targets)) / 2
            optim.zero_grad(); loss.backward(); optim.step(); sched.step()
            running += loss.item() * imgs.size(0)

        avg_loss = running / len(train_ds)
        metrics = retrieval_eval(model, val_dl, device)
        mtxt = ' '.join([f"{k}={v:.3f}" for k, v in metrics.items()])
        print(f"Epoch {ep:02d} | train_loss={avg_loss:.4f} | {mtxt}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save({
        'image_backbone': model.image_backbone.state_dict(),
        'text_backbone':  model.text_backbone.state_dict(),
        'img_proj':       model.img_proj.state_dict(),
        'txt_proj':       model.txt_proj.state_dict(),
        'logit_scale':    model.logit_scale.detach().cpu(),
        'text_model_name': args.text_model,
        'proj_dim': args.proj_dim,
    }, os.path.join(args.save_dir, 'clip_pretrain.pt'))
    print(f"Saved to {os.path.join(args.save_dir, 'clip_pretrain.pt')}")

if __name__ == '__main__':
    main()
