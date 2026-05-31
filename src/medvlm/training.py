from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from medvlm.config import MedVLMConfig
from medvlm.model import MedVLM
from medvlm.tokenizer import ReportTokenizer


Criterion = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class Trainer:
    def __init__(
        self,
        model: MedVLM,
        tokenizer: ReportTokenizer,
        config: MedVLMConfig,
        output_dir: str | Path = "outputs",
        criterion: Criterion | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.criterion = criterion

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=2000,
            eta_min=1e-6,
        )
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, loader) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True)
            reports = batch["report_ids"].to(self.device, non_blocking=True)
            inputs = reports[:, :-1]
            targets = reports[:, 1:]

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(inputs, images)
                loss = self._loss(logits, targets)
                scaled_loss = loss / self.config.grad_accum_steps

            self.scaler.scale(scaled_loss).backward()
            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self._optimizer_step()

            total_loss += loss.item()

        if len(loader) % self.config.grad_accum_steps != 0:
            self._optimizer_step()

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        total_loss = 0.0
        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            reports = batch["report_ids"].to(self.device, non_blocking=True)
            inputs = reports[:, :-1]
            targets = reports[:, 1:]

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(inputs, images)
                loss = self._loss(logits, targets)

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    def save_if_best(self, val_loss: float, name: str | None = None) -> Path | None:
        if val_loss >= self.best_val_loss:
            return None

        self.best_val_loss = val_loss
        filename = name or f"best_model_{self.config.encoder_type}.pt"
        path = self.output_dir / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.to_dict(),
                "tokenizer": {
                    "encoding": self.tokenizer.encoding_name,
                    "max_len": self.tokenizer.max_len,
                    "pad_id": self.tokenizer.pad_id,
                    "bos_id": self.tokenizer.bos_id,
                    "eos_id": self.tokenizer.eos_id,
                    "vocab_size": self.tokenizer.vocab_size,
                },
                "step": self.global_step,
                "best_val_loss": self.best_val_loss,
            },
            path,
        )
        return path

    def _optimizer_step(self) -> None:
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.global_step += 1

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.reshape(-1, self.tokenizer.vocab_size)
        flat_targets = targets.reshape(-1)
        if self.criterion is not None:
            return self.criterion(flat_logits, flat_targets)
        return F.cross_entropy(flat_logits, flat_targets, ignore_index=self.tokenizer.pad_id)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[MedVLM, ReportTokenizer, MedVLMConfig, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    raw_config = checkpoint.get("config", {})
    if isinstance(raw_config, dict):
        config = MedVLMConfig.from_dict(raw_config)
    else:
        config = MedVLMConfig.from_dict(
            {
                field: getattr(raw_config, field)
                for field in MedVLMConfig.__dataclass_fields__
                if hasattr(raw_config, field)
            }
        )
    tokenizer_info = checkpoint.get("tokenizer", {})
    tokenizer = ReportTokenizer(
        encoding=tokenizer_info.get("encoding", "gpt2"),
        max_len=tokenizer_info.get("max_len", config.max_len),
    )
    config.pretrained_backbone = False
    model = MedVLM(config, tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)
    state_dict = checkpoint.get("model") or checkpoint.get("model_state_dict")
    model.load_state_dict(state_dict)
    return model, tokenizer, config, checkpoint
