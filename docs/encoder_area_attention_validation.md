# Encoder-Only Validation Plan for Hybrid Area Attention

## Scope

This phase is encoder-only.

It does not claim end-to-end V-JEPA 2 gains, improved JEPA learning dynamics, or better world-model performance. The goal is to validate whether a hybrid local/global encoder retrofit is:

1. parameter-compatible with baseline encoder checkpoints,
2. forward-correct,
3. faster or more memory-efficient in the regimes that matter,
4. close enough in representation space to justify frozen-weight downstream checks.

## Claims, Separated

### 1. Parameter Compatibility

Claim: a baseline full-attention RoPE encoder state dict can load into the hybrid encoder strictly when the backbone hyperparameters match.

What is verified:

- Baseline encoder `state_dict()` loads into hybrid encoder with `strict=True`.
- No missing keys.
- No unexpected keys.

What is not implied:

- Arbitrary training checkpoints from different wrappers or configs always load strictly.
- Hub checkpoints that contain unrelated `pos_embed` entries or wrapper prefixes may still require cleanup or `strict=False` at the outer loading layer.

### 2. Forward-Pass Correctness

Claim: the hybrid encoder is functionally well-formed.

What is verified:

- Area-attention weights match RoPE attention parameter names.
- Single-area mode matches full attention numerically.
- Hybrid stacks build the intended area/global layer schedule.
- Sparse masked forward passes preserve expected tensor shapes.

What is not implied:

- Similarity of learned representations.
- Similar downstream task accuracy.

### 3. Runtime and Memory Scaling

Claim: hybrid area attention helps most when visible token count is large, especially in long-context, lightly masked, or fully visible encoder inference.

Benchmark regimes:

- `training_like`: 25% visible tokens.
- `lightly_masked`: 75% visible tokens.
- `fully_visible`: 100% visible tokens.

Core sweeps:

- Schedules: `full`, `12/12`, `18/6`, `20/4`, `area_only`.
- Frames: at least `16`, `32`, `64`.
- Outputs per row: schedule, regime, num frames, total tokens, visible tokens, latency, throughput, peak memory.

Interpretation rule:

- The main hypothesis is strongest for long-context, fully visible or lightly masked encoder inference.
- Masked pretraining-like regimes are still useful, but are secondary evidence.

### 4. Representation Retention

Claim: hybrid schedules preserve baseline encoder representations well enough to justify downstream frozen evaluation.

Cheap protocol:

1. Load the same pretrained checkpoint into a baseline encoder.
2. Copy the baseline encoder state dict into each hybrid schedule with `strict=True`.
3. Run the same clip batch through baseline and hybrid encoders.
4. Report:
   - mean token cosine similarity,
   - mean pooled cosine similarity,
   - linear CKA.

Suggested schedules:

- `12/12`
- `18/6`
- `20/4`

Optional downstream follow-up:

- Frozen video-classification evaluation for `12/12`, `18/6`, and `20/4`.

## Failure Modes and Overheads

Potential overheads that can erase theoretical wins:

- token sorting and reordering,
- partition construction from sparse visible-token indices,
- gather/scatter overhead at small sequence lengths,
- low-mask or tiny-input regimes where quadratic attention is not yet dominant,
- MLP becoming the runtime bottleneck after attention is reduced,
- excessive localization in late layers, which may reduce representation retention.

Expected qualitative behavior:

- High-mask training-like regimes may show smaller gains because visible token count is already reduced.
- Lightly masked and fully visible regimes should show the strongest memory and latency wins.
- Small inputs may not benefit because locality bookkeeping can dominate.
