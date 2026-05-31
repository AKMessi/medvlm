from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from medvlm.data import IUXRayDataset, prepare_iu_xray_split
from medvlm.evaluation import compute_length_stats, compute_term_presence, generate_report
from medvlm.training import load_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MedVLM checkpoint on validation samples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--output-csv", default="outputs/validation_predictions.csv")
    parser.add_argument("--min-gen-len", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, config, checkpoint = load_model_from_checkpoint(args.checkpoint, map_location=device)
    if args.data_root:
        config.data_root = args.data_root

    model.to(device).eval()
    _, val_df = prepare_iu_xray_split(config)
    val_df = val_df.head(args.max_samples).reset_index(drop=True)
    dataset = IUXRayDataset(val_df, tokenizer, config, is_train=False)

    rows = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0)
        pred = generate_report(model, tokenizer, image, min_gen_len=args.min_gen_len)
        ref = str(val_df.iloc[idx]["report_text"]).lower()
        rows.append(
            {
                "uid": sample["uid"],
                "image_path": sample["image_path"],
                "predicted": pred,
                "ground_truth": ref,
            }
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(output_path, index=False)

    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Wrote predictions: {output_path}")
    print(compute_length_stats(results["predicted"].tolist(), results["ground_truth"].tolist()))
    print(compute_term_presence(results["predicted"].tolist(), results["ground_truth"].tolist()).to_string(index=False))


if __name__ == "__main__":
    main()
