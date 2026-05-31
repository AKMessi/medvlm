from __future__ import annotations

import torch
import torch.nn.functional as F

from medvlm.config import MedVLMConfig


def average_cross_attention(attentions: list[torch.Tensor]) -> torch.Tensor:
    """Average cross-attention across decoder layers and heads for the last token."""

    if not attentions:
        raise ValueError("No attention tensors were returned by the model")
    stacked = torch.stack(attentions)
    if stacked.dim() != 5:
        raise ValueError(f"Expected attention shape [layers, batch, heads, text, image], got {stacked.shape}")
    return stacked.mean(dim=(0, 2))[0, -1, :]


def attention_to_grid(attention: torch.Tensor, config: MedVLMConfig) -> torch.Tensor:
    """Convert image-token attention into a square image grid."""

    if config.encoder_type == "resnet":
        patch_attention = attention
    else:
        patch_attention = attention[1:]

    grid_size = int(patch_attention.numel() ** 0.5)
    if grid_size * grid_size != patch_attention.numel():
        raise ValueError(f"Cannot reshape {patch_attention.numel()} image tokens into a square grid")
    return patch_attention.reshape(grid_size, grid_size)


def upsample_attention_grid(grid: torch.Tensor, img_size: int) -> torch.Tensor:
    """Upsample a low-resolution attention grid to image resolution."""

    return F.interpolate(
        grid.float().unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
