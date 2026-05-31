from __future__ import annotations

from collections import Counter

import pandas as pd
import torch

from medvlm.model import MedVLM
from medvlm.tokenizer import ReportTokenizer


DEFAULT_MEDICAL_TERMS = [
    "normal",
    "cardiomegaly",
    "effusion",
    "pneumonia",
    "fracture",
    "opacity",
    "consolidation",
    "pneumothorax",
    "enlarged",
    "clear",
]


@torch.no_grad()
def generate_report(
    model: MedVLM,
    tokenizer: ReportTokenizer,
    image: torch.Tensor,
    max_gen_len: int = 120,
    min_gen_len: int = 0,
    temperature: float = 0.8,
    top_k: int | None = 50,
) -> str:
    device = next(model.parameters()).device
    generated = model.generate(
        image.to(device),
        max_gen_len=max_gen_len,
        min_gen_len=min_gen_len,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0].detach().cpu().tolist())


def compute_length_stats(predictions: list[str], references: list[str]) -> dict[str, float]:
    pred_lengths = [len(text.split()) for text in predictions]
    ref_lengths = [len(text.split()) for text in references]
    pred_avg = sum(pred_lengths) / max(1, len(pred_lengths))
    ref_avg = sum(ref_lengths) / max(1, len(ref_lengths))
    return {
        "avg_prediction_words": pred_avg,
        "avg_reference_words": ref_avg,
        "length_ratio": pred_avg / ref_avg if ref_avg else 0.0,
    }


def compute_term_presence(
    predictions: list[str],
    references: list[str],
    terms: list[str] | None = None,
) -> pd.DataFrame:
    """Compute sample-aligned term precision/recall style counts."""

    terms = terms or DEFAULT_MEDICAL_TERMS
    rows = []
    for term in terms:
        pred_has = [term in pred.lower() for pred in predictions]
        ref_has = [term in ref.lower() for ref in references]
        tp = sum(p and r for p, r in zip(pred_has, ref_has))
        fp = sum(p and not r for p, r in zip(pred_has, ref_has))
        fn = sum((not p) and r for p, r in zip(pred_has, ref_has))
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        rows.append(
            {
                "term": term,
                "support": support,
                "predicted_mentions": sum(pred_has),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows)


def keyword_overlap(prediction: str, reference: str) -> dict[str, int]:
    pred_terms = Counter(prediction.lower().split())
    ref_terms = Counter(reference.lower().split())
    overlap = pred_terms & ref_terms
    return {
        "keyword_overlap": sum(overlap.values()),
        "prediction_words": sum(pred_terms.values()),
        "reference_words": sum(ref_terms.values()),
    }
