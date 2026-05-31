from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from medvlm.tokenizer import ReportTokenizer


PATHOLOGY_TOKEN_WEIGHTS = {
    "cardiomegaly": 10.0,
    "pneumonia": 10.0,
    "pneumothorax": 8.0,
    "effusion": 5.0,
    "opacity": 5.0,
    "consolidation": 5.0,
    "edema": 8.0,
    "normal": 0.5,
    "clear": 0.5,
}

PATHOLOGY_SAMPLE_KEYWORDS = [
    "cardiomegaly",
    "pneumonia",
    "pneumothorax",
    "effusion",
    "opacity",
    "consolidation",
    "edema",
    "mass",
    "nodule",
]


def create_pathology_token_weights(
    tokenizer: ReportTokenizer,
    weights: dict[str, float] | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    weights = weights or PATHOLOGY_TOKEN_WEIGHTS
    token_weights = torch.ones(tokenizer.vocab_size)
    for word, value in weights.items():
        for variant in (word, f" {word}"):
            for token_id in tokenizer.enc.encode(variant):
                if token_id < tokenizer.vocab_size:
                    token_weights[token_id] = value

    token_weights[tokenizer.pad_id] = 0.0
    token_weights[tokenizer.bos_id] = 1.0
    token_weights[tokenizer.eos_id] = 1.0
    if device is not None:
        token_weights = token_weights.to(device)
    return token_weights


class FocalCrossEntropy(nn.Module):
    """Focal cross entropy for rare-token emphasis."""

    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 1.5, ignore_index: int = -100):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = targets != self.ignore_index
        ce_loss = ce_loss[valid]
        if ce_loss.numel() == 0:
            return logits.sum() * 0
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def create_pathology_sampler(df, pathology_weight: float = 3.0) -> WeightedRandomSampler:
    sample_weights: list[float] = []
    for _, row in df.iterrows():
        text = f"{row.get('findings', '')} {row.get('impression', '')}".lower()
        sample_weights.append(pathology_weight if any(k in text for k in PATHOLOGY_SAMPLE_KEYWORDS) else 1.0)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
