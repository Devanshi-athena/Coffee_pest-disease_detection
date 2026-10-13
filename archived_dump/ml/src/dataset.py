"""Dataset and DataLoader helpers over data/processed/metadata.csv."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CoffeeLeafDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, classes: list[str], transform=None):
        self.paths = [Path(image_root) / p for p in df["path"]]
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.targets = [self.class_to_idx[c] for c in df["label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[i]


def load_metadata(processed_dir: Path, classes: list[str]) -> pd.DataFrame:
    csv = Path(processed_dir) / "metadata.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} not found. Build the dataset (ml/scripts/07_build_phase1_dataset.py) and upload the zip.")
    df = pd.read_csv(csv)
    unknown = set(df["label"]) - set(classes)
    if unknown:
        raise ValueError(f"metadata.csv has labels not in config classes: {sorted(unknown)}")
    return df


def class_weights(df: pd.DataFrame, classes: list[str]) -> torch.Tensor:
    """Inverse-frequency weights, normalized to mean 1. Missing classes get weight 0."""
    counts = df["label"].value_counts().reindex(classes, fill_value=0).to_numpy(dtype="float64")
    w = torch.tensor([1.0 / c if c else 0.0 for c in counts])
    return (w / w[w > 0].mean()).float()


def make_loader(df, image_root, classes, transform, batch_size, shuffle, num_workers) -> DataLoader:
    ds = CoffeeLeafDataset(df, image_root, classes, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False)
