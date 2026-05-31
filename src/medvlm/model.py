from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.models import ResNet50_Weights, resnet50

from medvlm.config import MedVLMConfig


def _resnet_weights(pretrained: bool):
    return ResNet50_Weights.DEFAULT if pretrained else None


class ViTEncoder(nn.Module):
    def __init__(self, config: MedVLMConfig):
        super().__init__()
        if config.img_size % config.vit_patch_size != 0:
            raise ValueError("img_size must be divisible by vit_patch_size")

        self.num_patches = (config.img_size // config.vit_patch_size) ** 2
        dim = config.vit_dim
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=config.vit_patch_size, stride=config.vit_patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.vit_heads,
            dim_feedforward=config.vit_mlp_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.vit_depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        x = self.patch_embed(image)
        x = rearrange(x, "b d h w -> b (h w) d")
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.dropout(x + self.pos_embed)
        return self.norm(self.transformer(x))


class ResNetEncoder(nn.Module):
    def __init__(self, config: MedVLMConfig):
        super().__init__()
        d_model = config.resolved_d_model()
        backbone = resnet50(weights=_resnet_weights(config.pretrained_backbone))
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.proj = nn.Linear(2048, d_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.repeat(1, 3, 1, 1)
        feats = self.backbone(image)
        feats = rearrange(feats, "b c h w -> b (h w) c")
        return self.proj(feats)


class HybridEncoder(nn.Module):
    """ResNet-50 local features followed by a compact transformer encoder."""

    def __init__(self, config: MedVLMConfig):
        super().__init__()
        d_model = config.resolved_d_model()
        backbone = resnet50(weights=_resnet_weights(config.pretrained_backbone))
        self.stem = nn.Sequential(*list(backbone.children())[:7])
        self.patch_proj = nn.Conv2d(1024, d_model, kernel_size=2, stride=2)

        grid_size = max(1, config.img_size // 32)
        self.num_patches = grid_size * grid_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.resolved_n_heads(),
            dim_feedforward=4 * d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        x = image.repeat(1, 3, 1, 1)
        x = self.stem(x)
        x = self.patch_proj(x)
        x = rearrange(x, "b d h w -> b (h w) d")

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        if x.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"Hybrid encoder produced {x.shape[1]} tokens, expected {self.pos_embed.shape[1]}. "
                "Use an img_size divisible by 32."
            )
        return self.transformer(x + self.pos_embed)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(
        self,
        text_feats: torch.Tensor,
        image_feats: torch.Tensor,
        causal_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, _ = self.self_attn(
            text_feats,
            text_feats,
            text_feats,
            attn_mask=causal_mask,
            need_weights=False,
        )
        text_feats = self.ln1(text_feats + attn_out)

        attn_out, attn_weights = self.cross_attn(
            text_feats,
            image_feats,
            image_feats,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        text_feats = self.ln2(text_feats + attn_out)
        text_feats = self.ln3(text_feats + self.ffn(text_feats))
        return text_feats, attn_weights


class MedVLM(nn.Module):
    def __init__(
        self,
        config: MedVLMConfig,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

        d_model = config.resolved_d_model()
        n_heads = config.resolved_n_heads()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        if config.encoder_type == "vit":
            self.encoder = ViTEncoder(config)
        elif config.encoder_type == "resnet":
            self.encoder = ResNetEncoder(config)
        elif config.encoder_type == "hybrid":
            self.encoder = HybridEncoder(config)
        else:
            raise ValueError(f"Unknown encoder_type: {config.encoder_type}")

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(config.max_len, d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, config.dropout) for _ in range(config.n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_len, config.max_len), diagonal=1).bool(),
        )

        self.apply(self._init_weights)
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        report_ids: torch.Tensor,
        image: torch.Tensor,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size, seq_len = report_ids.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.config.max_len}")

        image_feats = self.encoder(image)
        positions = torch.arange(seq_len, device=report_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.dropout(self.token_emb(report_ids) + self.pos_emb(positions))
        causal_mask = self.causal_mask[:seq_len, :seq_len]

        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            x, attn = block(x, image_feats, causal_mask, return_attention=return_attentions)
            if return_attentions and attn is not None:
                attentions.append(attn)

        logits = self.lm_head(x)
        if return_attentions:
            return logits, attentions
        return logits

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        max_gen_len: int = 120,
        min_gen_len: int = 0,
        temperature: float = 0.8,
        top_k: int | None = 50,
    ) -> torch.Tensor:
        self.eval()
        batch_size = image.shape[0]
        device = image.device
        generated = torch.full((batch_size, 1), self.bos_id, dtype=torch.long, device=device)

        for step in range(max_gen_len):
            logits = self.forward(generated, image)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_logits = logits[:, -1, :]
            if step < min_gen_len:
                next_logits[:, self.eos_id] = -float("inf")

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
                    next_logits = next_logits.masked_fill(next_logits < values[:, [-1]], -float("inf"))
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == self.eos_id).all():
                break
        return generated
