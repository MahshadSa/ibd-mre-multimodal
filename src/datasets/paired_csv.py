from pathlib import Path
from typing import Optional, Callable
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class PairedImageTextCSV(Dataset):
    """CSV with columns: image_path, text."""
    def __init__(self, csv_path: str, transform: Optional[Callable]=None, text_proc: Optional[Callable]=None):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)
        assert {'image_path','text'}.issubset(self.df.columns)
        self.root = self.csv_path.parent
        self.transform = transform
        self.text_proc = text_proc

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])
        if not img_path.is_absolute(): img_path = self.root / img_path
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        text = str(row['text'])
        if self.text_proc: text = self.text_proc(text)
        return img, text
