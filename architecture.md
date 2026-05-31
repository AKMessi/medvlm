# MedVLM Architecture

This document explains the model from first principles, using the implementation in `src/medvlm/model.py` as the source of truth.

MedVLM is a chest X-ray report generator. It takes one frontal chest X-ray image and generates a radiology-style text report one token at a time.

At the highest level:

```text
Chest X-ray image
    -> image preprocessing
    -> visual encoder
    -> visual tokens
    -> transformer text decoder with image cross-attention
    -> next-token probabilities
    -> generated report
```

The important idea is simple: the image is converted into a small sequence of "visual words", and the report decoder writes text while repeatedly looking back at those visual words.

## One-Screen Summary

Default configuration:

```text
Input image:
    [B, 1, 224, 224]

Hybrid visual encoder:
    ResNet-50 stem/layers -> [B, 1024, 14, 14]
    2x2 stride-2 conv     -> [B, 512, 7, 7]
    flatten patches       -> [B, 49, 512]
    add CLS token         -> [B, 50, 512]
    4-layer transformer   -> [B, 50, 512]

Text decoder:
    report tokens         -> [B, T]
    token + pos embedding -> [B, T, 512]
    6 decoder blocks      -> [B, T, 512]
    vocabulary head       -> [B, T, vocab_size]

Output:
    next-token logits for report generation
```

The default model is roughly 74M parameters in the original notebook run.

## Why This Architecture Exists

Chest X-ray report generation is not just image classification. A classifier asks:

```text
"Is there pneumonia?"
```

A report generator asks:

```text
"What should be said, in what order, with what level of certainty?"
```

That means the model needs two abilities:

1. Visual understanding: it must encode anatomy and image findings.
2. Language generation: it must produce coherent report text token by token.

MedVLM splits those responsibilities:

- The visual encoder turns the X-ray into image tokens.
- The text decoder turns previous report tokens plus image tokens into the next report token.

This is the same broad idea behind many vision-language models: image features become a sequence, text generation attends to that sequence.

## Data Flow

The dataset loader joins projection metadata and reports by `uid`, filters frontal images, and builds report text from findings plus impression.

```text
indiana_projections.csv
indiana_reports.csv
        |
        v
merge on uid
        |
        v
filter projection == "Frontal"
        |
        v
image_path + report_text
```

The split is done by `uid`, not by row. That matters because a single study can have related images/reports. Splitting by UID helps avoid train/validation leakage.

## Image Preprocessing

The image pipeline uses MONAI transforms:

```text
LoadImage(image_only=True)
EnsureChannelFirst()
ScaleIntensity()
Resize((224, 224))
```

The final tensor shape is:

```text
[B, 1, 224, 224]
```

That is batch, channel, height, width. The channel count is 1 because chest X-rays are grayscale.

During training, the dataset can also apply small rotations and zooms. Horizontal flipping is disabled by default in the cleaned repo because left/right anatomy in radiology can matter.

## Tokenization

Reports are tokenized with GPT-2 BPE through `tiktoken`.

The project adds three explicit special tokens:

```text
PAD_ID = gpt2_vocab_size
BOS_ID = gpt2_vocab_size + 1
EOS_ID = gpt2_vocab_size + 2
```

Each report becomes:

```text
[BOS] + report_tokens + [EOS] + padding
```

The default max length is 256 tokens.

Intuition:

- `BOS` tells the model "start writing now".
- `EOS` tells the model "the report is complete".
- `PAD` fills the batch so every report has the same length.

The loss ignores padding tokens, so the model is not punished for padded positions.

## Visual Encoder Options

The code supports three encoders:

1. `resnet`
2. `vit`
3. `hybrid`

The default and main notebook model is `hybrid`.

## Encoder 1: ResNet Encoder

The ResNet encoder uses ResNet-50 as a spatial feature extractor.

Input:

```text
[B, 1, 224, 224]
```

ResNet-50 expects 3 channels, so the grayscale X-ray is repeated:

```text
[B, 1, 224, 224] -> [B, 3, 224, 224]
```

The classifier head is removed. The model keeps the convolutional feature map:

```text
ResNet-50 without avgpool/fc -> [B, 2048, 7, 7]
```

Then the spatial grid is flattened:

```text
[B, 2048, 7, 7] -> [B, 49, 2048]
```

Finally, a linear projection maps each patch from 2048 dimensions to the decoder dimension:

```text
[B, 49, 2048] -> [B, 49, d_model]
```

Intuition: each of the 49 tokens represents a coarse region of the image.

## Encoder 2: Pure ViT Encoder

The ViT encoder directly patchifies the image.

For a 224 x 224 image and a patch size of 16:

```text
224 / 16 = 14
14 * 14 = 196 patches
```

The patch embedding is implemented as a convolution:

```text
Conv2d(1, 768, kernel_size=16, stride=16)
```

Shape flow:

```text
[B, 1, 224, 224]
    -> [B, 768, 14, 14]
    -> [B, 196, 768]
    -> add CLS token
    -> [B, 197, 768]
    -> transformer encoder
    -> [B, 197, 768]
```

Intuition: instead of starting with CNN features, the image is treated almost like a sentence of 196 image patches.

## Encoder 3: Hybrid Encoder

The hybrid encoder is the main model.

It combines:

- CNN inductive bias for local image patterns.
- Transformer attention for relationships across image regions.

The flow:

```text
Input image:
    [B, 1, 224, 224]

Repeat grayscale channel:
    [B, 3, 224, 224]

ResNet-50 through layer3:
    [B, 1024, 14, 14]

Patch projection:
    Conv2d(1024, 512, kernel_size=2, stride=2)
    [B, 1024, 14, 14] -> [B, 512, 7, 7]

Flatten:
    [B, 512, 7, 7] -> [B, 49, 512]

Add CLS token:
    [B, 49, 512] -> [B, 50, 512]

Add learned positional embeddings:
    [B, 50, 512]

Transformer encoder:
    [B, 50, 512]
```

Why this is a good compromise:

- ResNet sees local edges, textures, lung fields, cardiac silhouette, and other spatial cues efficiently.
- The transformer encoder lets all 49 regions communicate globally.
- The final sequence is short: only 50 visual tokens. That keeps cross-attention manageable.

The image is not reduced to one vector. It stays as a grid of visual tokens, which is important because the decoder can attend to different parts of the X-ray while generating different words.

## The CLS Token

The hybrid and ViT encoders prepend a learned `CLS` token.

Shape:

```text
patch tokens: [B, 49, 512]
cls token:    [B, 1, 512]
combined:     [B, 50, 512]
```

Intuition:

- Patch tokens represent local regions.
- The CLS token can act as a global summary token.

The decoder is free to attend to local patch tokens or the global CLS token.

## Positional Embeddings

Transformers do not naturally know order or position. A patch token by itself does not say whether it came from the upper-left lung field or lower-right base.

So the encoder adds learned positional embeddings:

```text
visual_tokens + visual_position_embeddings
```

The decoder also uses learned positional embeddings for text tokens:

```text
token_embeddings + text_position_embeddings
```

Without positional embeddings, the model would know what features exist but not where they are or where each word sits in the report.

## Text Decoder

The decoder is a causal transformer language model with image cross-attention.

Input:

```text
report_ids: [B, T]
image:      [B, 1, 224, 224]
```

First, the image is encoded:

```text
image_feats = encoder(image)
image_feats: [B, S, D]
```

For the default hybrid encoder:

```text
S = 50
D = 512
```

Then the report token IDs are embedded:

```text
token_emb(report_ids) + pos_emb(positions)
    -> [B, T, 512]
```

That text sequence goes through `n_layers` decoder blocks. The default is 6.

## Decoder Block

Each decoder block has three stages:

```text
1. causal self-attention over text
2. cross-attention from text to image
3. feed-forward network
```

In code:

```text
x = LayerNorm(x + self_attention(x))
x = LayerNorm(x + cross_attention(x, image_feats))
x = LayerNorm(x + feed_forward(x))
```

The intuition:

1. Self-attention asks: "What have I already written?"
2. Cross-attention asks: "What part of the image is relevant now?"
3. Feed-forward layers refine the representation before the next block.

## Causal Self-Attention

During report generation, token 10 should not be allowed to look at token 11. That would leak the future.

The model uses a causal mask:

```text
position i can attend to positions <= i
position i cannot attend to positions > i
```

Example:

```text
Allowed:
token 4 -> tokens 1, 2, 3, 4

Blocked:
token 4 -> tokens 5, 6, 7, ...
```

This makes training match generation. The model learns to predict the next token using only previous text plus the image.

## Cross-Attention

Cross-attention is the key vision-language bridge.

In self-attention, text attends to text:

```text
queries = text
keys    = text
values  = text
```

In cross-attention, text attends to image:

```text
queries = text
keys    = image tokens
values  = image tokens
```

For each word position, the decoder can ask the image a question.

Example intuition:

```text
When writing "heart", attend to cardiac silhouette regions.
When writing "effusion", attend to costophrenic angle / lower lung regions.
When writing "pneumothorax", attend to pleural boundary regions.
```

The model is not explicitly told those regions. It learns this alignment from image-report pairs.

## Feed-Forward Network

After attention, each token passes through an MLP:

```text
Linear(512, 2048)
GELU
Dropout
Linear(2048, 512)
Dropout
```

Attention mixes information across tokens. The feed-forward network transforms each token's representation independently. Both are needed.

## Output Head

After the decoder blocks, the model projects hidden states to vocabulary logits:

```text
[B, T, 512] -> [B, T, vocab_size]
```

Each position gets a probability distribution over the next token.

The output head shares weights with the token embedding matrix:

```text
lm_head.weight = token_emb.weight
```

This is called weight tying. It reduces parameters and usually helps language models because the model uses the same token geometry for reading and writing.

## Training

Training uses teacher forcing.

A full encoded report looks like:

```text
[BOS, token_1, token_2, token_3, ..., EOS, PAD, PAD]
```

The model input is:

```text
[BOS, token_1, token_2, token_3, ...]
```

The target is shifted by one:

```text
[token_1, token_2, token_3, ..., EOS]
```

So the task is:

```text
given image + previous report tokens, predict the next report token
```

Loss:

```text
cross_entropy(logits, targets, ignore_index=PAD_ID)
```

Padding is ignored because padded positions are not real text.

## Optimization

The training script uses:

- AdamW optimizer.
- Weight decay.
- Gradient clipping.
- Optional mixed precision on CUDA.
- Gradient accumulation.
- Cosine warm restarts scheduler.

Gradient accumulation is important because image-language models can be memory heavy. For example:

```text
batch_size = 4
grad_accum_steps = 4
effective batch size = 16
```

That gives a larger effective batch without needing all 16 samples in GPU memory at the same time.

## Rare-Finding Improvements

Medical report datasets are imbalanced. Normal or common phrases appear often; rare pathology terms appear much less often.

That creates a bad shortcut:

```text
The model learns that "normal / clear / no acute disease" is often safe.
```

The repo includes optional tools to fight that:

1. Pathology token weights
2. Focal cross entropy
3. Pathology-balanced sampler

The token weighting raises loss for terms like:

```text
cardiomegaly
pneumonia
pneumothorax
effusion
opacity
consolidation
edema
```

The sampler gives studies containing pathology terms a higher chance of appearing in training batches.

These are not magic. They improve the training signal for rare findings, but a proper clinical evaluation is still required.

## Inference

At inference time, there is no ground-truth report. The model writes from scratch.

Generation starts with:

```text
[BOS]
```

Then the loop is:

```text
1. run model on current generated tokens + image
2. take logits from the last position
3. sample or argmax the next token
4. append that token
5. stop if EOS is generated
```

In code, this is `MedVLM.generate(...)`.

Controls:

- `temperature`: lower is more conservative, higher is more random.
- `top_k`: restricts sampling to the top-k likely tokens.
- `min_gen_len`: prevents the model from stopping too early.
- `max_gen_len`: hard cap on report length.

## Attention Maps

When `return_attentions=True`, each decoder block can return cross-attention weights.

Shape:

```text
[layers, B, heads, text_tokens, image_tokens]
```

For the hybrid encoder:

```text
image_tokens = 50 = 1 CLS + 49 patches
```

To visualize attention:

```text
1. average across layers and heads
2. take the last generated text token
3. remove CLS token
4. reshape 49 patch scores to 7 x 7
5. upsample to 224 x 224
6. overlay on the X-ray
```

Important caveat: attention maps are interpretability aids. They are not proof that the model is clinically reasoning correctly.

## Why The Default Is Hybrid

Pure CNN:

- Good at local visual patterns.
- Efficient.
- But global relations are less direct.

Pure ViT:

- Good at global patch relationships.
- Flexible.
- But typically wants more data and compute.

Hybrid:

- CNN handles low-level image structure.
- Transformer handles region-to-region relationships.
- Decoder gets a compact visual sequence.

For a small dataset like IU X-Ray, the hybrid setup is a pragmatic middle ground.

## What The Model Learns Well

From the notebook logs, the model learned:

- Report-like sentence structure.
- Common radiology phrasing.
- Basic image-conditioned generation patterns.
- Some improvement from rare-finding weighting and balanced sampling.

That makes it a good research and portfolio prototype.

## What The Model Does Not Yet Prove

The current repo does not claim clinical-grade performance.

Missing pieces for a stronger research claim:

- Locked train/validation/test split.
- BLEU, ROUGE, METEOR, CIDEr, or similar text metrics.
- Clinical label metrics using a report labeler such as CheXbert-style extraction.
- Entity/relation metrics such as RadGraph-style evaluation.
- External validation on a separate dataset.
- Human radiologist review.
- Calibration and uncertainty analysis.

The architecture is legitimate. The current results are exploratory.

## Mental Model

The easiest way to understand MedVLM:

```text
The encoder turns the X-ray into 50 visual memory slots.
The decoder writes a report one word-piece at a time.
At every word-piece, the decoder can look at:
    1. what it has already written
    2. the visual memory slots from the X-ray
```

That is the whole system.

Everything else is engineering around that core idea:

- Better visual tokens.
- Better text generation.
- Better loss for rare findings.
- Better evaluation.
- Better safety framing.
