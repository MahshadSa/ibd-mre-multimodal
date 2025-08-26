import argparse, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from src.datasets.seg_pairs_csv import SegPairsCSV
from src.branches.image import ResNet18Encoder
import torch.nn.functional as F
import numpy as np
from pathlib import Path

class LightDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.up32 = nn.Sequential(nn.Conv2d(512,256,3,padding=1), nn.ReLU(inplace=True))
        self.up16 = nn.Sequential(nn.Conv2d(256+256,128,3,padding=1), nn.ReLU(inplace=True))
        self.up8  = nn.Sequential(nn.Conv2d(128+128,64, 3,padding=1), nn.ReLU(inplace=True))
        self.up4  = nn.Sequential(nn.Conv2d(64+64,  64, 3,padding=1), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, f4, f8, f16, f32):
        x = F.interpolate(self.up32(f32), scale_factor=2, mode='bilinear', align_corners=False)  # /16
        x = torch.cat([x, f16], dim=1)
        x = F.interpolate(self.up16(x), scale_factor=2, mode='bilinear', align_corners=False)    # /8
        x = torch.cat([x, f8], dim=1)
        x = F.interpolate(self.up8(x),  scale_factor=2, mode='bilinear', align_corners=False)    # /4
        x = torch.cat([x, f4], dim=1)
        x = self.up4(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)               # /1
        return self.head(x)

class SegModel(nn.Module):
    def __init__(self, num_classes: int, in_channels: int=1, pretrained=True):
        super().__init__()
        self.encoder = ResNet18Encoder(in_channels=in_channels, pretrained=pretrained)
        self.decoder = LightDecoder(num_classes=num_classes)
    def forward(self, x):
        f4,f8,f16,f32 = self.encoder(x)
        return self.decoder(f4,f8,f16,f32)

@torch.no_grad()
def dice_score(logits: torch.Tensor, masks: torch.Tensor, num_classes: int, eps=1e-6):
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    dices = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        tgt_c  = (masks == c).float()
        inter = (pred_c*tgt_c).sum(dim=(1,2))
        denom = pred_c.sum(dim=(1,2)) + tgt_c.sum(dim=(1,2))
        dice_c = ((2*inter + eps) / (denom + eps)).mean().item()
        dices.append(dice_c)
    return float(np.mean(dices))

def train_one_epoch(model, dl, optim, loss_fn, scaler=None, device="cuda"):
    model.train(); running=0.0
    for imgs, masks in tqdm(dl, desc="Train", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device_type='cuda', enabled=(device=='cuda')):
            logits = model(imgs)
            loss = loss_fn(logits, masks)
        optim.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); scaler.step(optim); scaler.update()
        else:
            loss.backward(); optim.step()
        running += loss.item() * imgs.size(0)
    return running / len(dl.dataset)

@torch.no_grad()
def validate(model, dl, loss_fn, device="cuda", num_classes=11):
    model.eval(); running=0.0; dices=[]
    for imgs, masks in tqdm(dl, desc="Val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        running += loss.item() * imgs.size(0)
        dices.append(dice_score(logits, masks, num_classes))
    return running/len(dl.dataset), float(np.mean(dices)) if dices else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/mre_slices.csv")
    ap.add_argument("--num-classes", type=int, default=11)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save-dir", type=str, default="artifacts/mre_seg_pretrain")
    args = ap.parse_args()

    device = torch.device(args.device)
    ds = SegPairsCSV(args.csv, img_size=args.img_size, in_channels=args.in_channels, num_classes=args.num_classes)
    n_val = max(1, int(len(ds)*args.val_split)); n_tr = len(ds) - n_val
    tr_ds, va_ds = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SegModel(args.num_classes, args.in_channels, pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device=='cuda'))

    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "seg_best.pt"
    best_val = 1e9

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr, optim, loss_fn, scaler, device)
        va_loss, va_dice = validate(model, va, loss_fn, device, args.num_classes)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_dice={va_dice:.3f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
            print(f"  ↳ Saved best to {best_path}")

    # Export encoder for Stage 3
    enc_path = save_dir / "resnet18_mre_encoder.pt"
    tmp = SegModel(args.num_classes, args.in_channels, pretrained=False)
    tmp.load_state_dict(torch.load(best_path, map_location="cpu"), strict=True)
    torch.save(tmp.encoder.state_dict(), enc_path)
    print("Exported encoder to:", enc_path)

if __name__ == "__main__":
    main()
