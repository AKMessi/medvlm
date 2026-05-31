from __future__ import annotations

import argparse
from pathlib import Path

import torch

from medvlm.config import MedVLMConfig
from medvlm.data import create_dataloaders, prepare_iu_xray_split
from medvlm.losses import FocalCrossEntropy, create_pathology_sampler, create_pathology_token_weights
from medvlm.model import MedVLM
from medvlm.tokenizer import ReportTokenizer
from medvlm.training import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MedVLM on IU X-Ray.")
    parser.add_argument("--data-root", default="/kaggle/input/chest-xrays-indiana-university")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--encoder-type", choices=["resnet", "vit", "hybrid"], default="hybrid")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--loss", choices=["ce", "focal"], default="ce")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MedVLMConfig(
        data_root=args.data_root,
        encoder_type=args.encoder_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        pretrained_backbone=not args.no_pretrained,
        horizontal_flip_prob=args.horizontal_flip_prob,
    )

    tokenizer = ReportTokenizer(max_len=config.max_len)
    train_df, val_df = prepare_iu_xray_split(config)
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    sampler = create_pathology_sampler(train_df) if args.balanced_sampler else None
    train_loader, val_loader = create_dataloaders(train_df, val_df, tokenizer, config, train_sampler=sampler)

    model = MedVLM(config, tokenizer.vocab_size, tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)
    criterion = None
    if args.loss == "focal":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = create_pathology_token_weights(tokenizer, device=device)
        criterion = FocalCrossEntropy(weight=weights, gamma=1.5, ignore_index=tokenizer.pad_id)

    trainer = Trainer(model, tokenizer, config, output_dir=Path(args.output_dir), criterion=criterion)
    print(f"Device: {trainer.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        saved = trainer.save_if_best(val_loss)
        status = f" | saved {saved}" if saved else ""
        print(f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}{status}")


if __name__ == "__main__":
    main()
