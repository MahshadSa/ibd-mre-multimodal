from typing import Tuple
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

class SegPairsCSV(Dataset):
    """CSV with columns: image_path, mask_path """
    def __init__(self, csv_path: str, img_size=256, in_channels=1, num_classes=11):
        self.df = pd.read_csv(csv_path)
        assert {"image_path","mask_path"}.issubset(self.df.columns), "CSV must have image_path,mask_path"
        self.num_classes = num_classes
        self.img_tf = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.mask_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.NEAREST),
            T.PILToTensor(),  
        ])

    def __len__(self): return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.df.iloc[idx]
        img = Image.open(r["image_path"]).convert("L")
        msk = Image.open(r["mask_path"])
        img = self.img_tf(img)              
        msk = self.mask_tf(msk).squeeze(0)  
        msk = torch.clamp(msk.long(), 0, self.num_classes-1)
        return img, msk
