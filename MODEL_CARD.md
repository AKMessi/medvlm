# Model Card: MedVLM IU X-Ray

Repository: https://github.com/AKMessi/medvlm

Author: Aaryan Kakad ([AKMessi](https://github.com/AKMessi))

## Intended Use

MedVLM is a research prototype for chest X-ray report generation. It is intended for machine-learning portfolio demonstration, architecture exploration, and educational experiments.

It is not intended for clinical use, diagnosis, triage, or patient-care decisions.

## Model Details

- Architecture: hybrid ResNet-50 and transformer image encoder with a causal transformer decoder.
- Input: frontal chest X-ray image resized to 224 x 224.
- Output: free-text radiology-style report.
- Tokenizer: GPT-2 BPE with PAD/BOS/EOS special tokens.
- Dataset used in the notebook: IU X-Ray reports and images.

## Training Data

The notebook filters to frontal images and joins projection metadata with report text by UID. Images and labels are not committed to this repository.

## Evaluation

The recovered notebook includes validation-loss logs, generated report examples, attention visualizations, and quick pathology-term checks. These are exploratory checks only. A production-grade evaluation would require locked splits, report-level metrics, pathology-label extraction, calibration, expert review, and external validation.

## Limitations

- The model can hallucinate findings or omit clinically important findings.
- The dataset is small for modern medical report generation.
- Text reports contain missing values and noisy wording.
- Some notebook metrics are sampled heuristics rather than formal benchmarks.
- Attention maps are interpretability aids, not proof of clinical reasoning.

## Safety

Do not use this model for medical decisions. Any generated report must be treated as unverified synthetic text.

## License

The repository code is licensed under Apache-2.0. Dataset files, X-ray images, radiology reports, and patient-derived qualitative examples are subject to the original IU/Open-i dataset terms and are not covered by the code license.
