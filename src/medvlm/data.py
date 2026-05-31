from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandRotate,
    RandZoom,
    Resize,
    ScaleIntensity,
)
from torch.utils.data import DataLoader, Dataset, Sampler

from medvlm.config import MedVLMConfig
from medvlm.tokenizer import ReportTokenizer


def load_iu_xray_dataframe(config: MedVLMConfig, require_images: bool = True) -> pd.DataFrame:
    """Load and join the IU X-Ray projection/report CSV files."""

    root = Path(config.data_root)
    projections_path = root / "indiana_projections.csv"
    reports_path = root / "indiana_reports.csv"

    projections = pd.read_csv(projections_path)
    reports = pd.read_csv(reports_path)

    df = projections.merge(reports, how="inner", on="uid")
    if config.projection:
        df = df[df["projection"] == config.projection]

    image_root = root / config.image_dir
    df["image_path"] = df["filename"].apply(lambda name: str(image_root / str(name)))
    if require_images:
        df = df[df["image_path"].apply(os.path.exists)]

    findings = df["findings"].fillna("").astype(str)
    impression = df["impression"].fillna("").astype(str)
    df["report_text"] = ("findings: " + findings + " impression: " + impression).str.strip()
    return df.reset_index(drop=True)


def split_by_uid(
    df: pd.DataFrame,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by study UID to prevent image/report leakage across splits."""

    uids = list(df["uid"].unique())
    rng = random.Random(seed)
    rng.shuffle(uids)

    split_idx = int((1.0 - val_fraction) * len(uids))
    train_uids = set(uids[:split_idx])
    val_uids = set(uids[split_idx:])

    leakage = train_uids.intersection(val_uids)
    if leakage:
        raise ValueError(f"UID leakage found between train and validation: {sorted(leakage)[:5]}")

    train_df = df[df["uid"].isin(train_uids)].reset_index(drop=True)
    val_df = df[df["uid"].isin(val_uids)].reset_index(drop=True)
    return train_df, val_df


def prepare_iu_xray_split(config: MedVLMConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_iu_xray_dataframe(config)
    return split_by_uid(df, val_fraction=config.val_fraction, seed=config.seed)


def build_image_transform(config: MedVLMConfig, is_train: bool) -> Compose:
    transforms = [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize((config.img_size, config.img_size)),
    ]
    if is_train:
        transforms.extend(
            [
                RandRotate(range_x=0.1, prob=config.rotate_prob),
                RandZoom(min_zoom=0.95, max_zoom=1.05, prob=config.zoom_prob),
            ]
        )
        if config.horizontal_flip_prob > 0:
            from monai.transforms import RandFlip

            transforms.append(RandFlip(spatial_axis=1, prob=config.horizontal_flip_prob))
    return Compose(transforms)


class IUXRayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: ReportTokenizer, config: MedVLMConfig, is_train: bool):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = build_image_transform(config, is_train=is_train)
        self.img_size = config.img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        try:
            image = self.transform(row["image_path"])
        except Exception:
            image = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        return {
            "image": image,
            "report_ids": self.tokenizer.encode(row["report_text"]),
            "uid": row.get("uid", idx),
            "image_path": row["image_path"],
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer: ReportTokenizer,
    config: MedVLMConfig,
    train_sampler: Sampler | None = None,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = IUXRayDataset(train_df, tokenizer, config, is_train=True)
    val_dataset = IUXRayDataset(val_df, tokenizer, config, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
