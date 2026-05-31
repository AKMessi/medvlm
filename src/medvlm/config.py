from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


EncoderType = Literal["resnet", "vit", "hybrid"]


@dataclass
class MedVLMConfig:
    """Configuration shared by data loading, training, and inference."""

    data_root: str = "/kaggle/input/chest-xrays-indiana-university"
    image_dir: str = "images/images_normalized"
    projection: str = "Frontal"
    seed: int = 42
    val_fraction: float = 0.10

    img_size: int = 224
    max_len: int = 256
    batch_size: int = 4
    num_workers: int = 2

    encoder_type: EncoderType = "hybrid"
    pretrained_backbone: bool = True

    vit_patch_size: int = 16
    vit_dim: int = 768
    vit_depth: int = 12
    vit_heads: int = 12
    vit_mlp_dim: int = 3072

    d_model: int | None = None
    n_heads: int | None = None
    n_layers: int = 6
    dropout: float = 0.10

    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_accum_steps: int = 4
    mixed_precision: bool = True

    # Keep this disabled by default because left/right flips can contradict reports.
    horizontal_flip_prob: float = 0.0
    rotate_prob: float = 0.3
    zoom_prob: float = 0.3

    def resolved_d_model(self) -> int:
        if self.d_model is not None:
            return self.d_model
        return self.vit_dim if self.encoder_type == "vit" else 512

    def resolved_n_heads(self) -> int:
        if self.n_heads is not None:
            return self.n_heads
        return self.vit_heads if self.encoder_type == "vit" else 8

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "MedVLMConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in allowed})
