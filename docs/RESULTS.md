# Results Notes

This page documents what was recovered from the original notebook output. The results are useful for showcasing the project, but they are not a replacement for a formal radiology report-generation benchmark.

## Dataset And Split

- Dataset: Indiana University chest X-ray report dataset from Kaggle.
- Filter: frontal projections only.
- Logged frontal rows: 3,818.
- Final hybrid training split: 3,435 train samples and 383 validation samples.
- Split type: UID-level split with no train/validation UID overlap.

## Model

- Encoder: hybrid ResNet-50 + transformer image encoder.
- Image tokens: 49 patches plus one CLS token for the hybrid encoder.
- Decoder: 6-layer causal transformer decoder with image cross-attention.
- Tokenizer: GPT-2 BPE plus PAD/BOS/EOS tokens.
- Logged parameter count: 74.36M to 74.4M.

## Logged Training

The main hybrid run logged:

| Epoch | Train loss | Validation loss |
| --- | ---: | ---: |
| 1 | 2.8409 | 1.7759 |
| 2 | 1.5207 | 1.5145 |
| 3 | 1.2959 | 1.3923 |
| 4 | 1.0222 | 1.2917 |
| 6 | 0.9235 | 1.2804 |
| 7 | 0.8444 | 1.2653 |
| 8 | 0.7876 | 1.2634 |
| 9 | 0.7575 | 1.2631 |
| 10 | 0.8529 | 1.3253 |

The epoch numbering in the notebook output skips epoch 5, likely from a rerun/interruption in the old Kaggle session. The best logged validation loss was 1.2631.

## Evaluation Notes

The notebook's baseline 20-sample evaluation logged:

- Average prediction length: 27.5 words.
- Average ground-truth length: 37.9 words.
- Length ratio: 0.73.
- Medical term checks: normal 19/17, cardiomegaly 0/2, effusion 8/17, pneumonia 0/2.

The later pathology-focused continuation logged:

- Effusion: 24/28.
- Pneumothorax: 11/22 in the final sampled check, with an earlier sampled check logging 17/22.
- Pneumonia: 1/2.
- Opacity: 1/5.
- Cardiomegaly follow-up: 7/10 cardiac-term detections on a sampled set of cardiomegaly validation cases.

Important caveat: the original notebook mixed several quick evaluation heuristics. The cleaned repository implements sample-aligned term counts in `src/medvlm/evaluation.py`.

## Figures

- `assets/results/architecture.png`: architecture overview.
- `assets/results/metrics_summary.png`: notebook-generated metrics graphic.
- `assets/results/before_after.png`: summary of the notebook's rare-finding mitigation experiment.

Patient-derived attention maps, X-ray overlays, and qualitative report examples are not committed to this public repository. They can be regenerated locally after downloading the IU/Open-i dataset under its original CC BY-NC-ND 4.0 terms.
