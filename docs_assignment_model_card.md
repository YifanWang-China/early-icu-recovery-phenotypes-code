# Fixed-label temporal assignment model card

This model card summarizes the temporal assignment model used to map admissions to the frozen discovery recovery phenotypes. It is intended to make the external assignment step auditable for co-authors, reviewers and future repository deposition.

## Intended role

The assignment model maps postoperative laboratory trajectories to probabilities for the fixed discovery phenotypes S0, S1 and S2. It is used for internal temporal holdout assignment, external Korean assignment and external MIMIC-IV stress-test assignment. It does not re-cluster external data and does not define new local phenotypes.

## Input representation

| Component | Specification |
| --- | --- |
| Time window | Postoperative 0-6 h |
| Temporal grid | 13 reference times, every 0.5 h |
| Shared laboratory channels | Lactate, pH, PaO2, PaCO2, hemoglobin, potassium, sodium and glucose |
| Value preprocessing | Standardized values |
| Missing values | Filled with zero after standardization |
| Missingness representation | Binary missingness/observation mask concatenated with values |
| Per-time-step input width | 16 features, comprising 8 standardized values and 8 missingness indicators |

## Architecture

| Parameter | Value |
| --- | --- |
| Model family | Transformer encoder |
| Input projection | Linear projection to latent dimension |
| Latent dimension | d_model = 64 |
| Positional encoding | Learned positional embeddings |
| Transformer layers | 3 encoder layers |
| Attention heads | 4 |
| Feed-forward dimension | 256 |
| Activation | GELU |
| Dropout | 0.1 |
| Tensor layout | Batch-first |
| Sequence aggregation | Layer normalization followed by mean pooling |
| Output heads | Phenotype classifier and value-reconstruction decoder |

## Training setup

| Parameter | Value |
| --- | --- |
| Supervision target | Frozen discovery phenotype labels |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 256 in the locked fixed-label transport run |
| Epochs | 40 in the locked fixed-label transport run |
| Random seed | 42 |
| Reconstruction target | Observed laboratory values |
| Augmentation | Random channel dropout for augmented reconstruction |
| Consistency term | Cosine consistency between original and augmented embeddings |
| Locked classification weight | alpha = 2.0 |
| Locked consistency weight | beta = 0.2 |
| Development variants | alpha = 1.0 and strengthened phenotype-supervision variants were evaluated during development |

## Loss components

The training objective combines four components:

1. Observed-value reconstruction loss.
2. Augmented reconstruction loss after random channel dropout.
3. Phenotype classification loss against frozen discovery labels.
4. Cosine embedding-consistency loss between original and augmented inputs.

## Assignment output and confidence

The classifier produces probabilities for S0, S1 and S2. The assigned phenotype is the class with the largest probability. Assignment confidence is summarized as the maximum assignment probability. The current manuscript reports external assignment-confidence distributions rather than using confidence as a treatment threshold.

## Validation and transport boundaries

| Boundary | Current handling |
| --- | --- |
| No external re-clustering | External cohorts are mapped to frozen discovery phenotypes |
| Temporal holdout | Late holdout admissions are not used for discovery or branch selection |
| Missingness | Missingness masks are explicit model inputs |
| Incomplete early data | Evaluated by cumulative-window and missingness stress tests |
| Morphology equivalence | Evaluated separately by dynamic-time-warping equivalence analysis |
| Clinical deployment | Not validated as a treatment rule |
| Laboratory ordering | Missingness handling does not imply fewer tests should be ordered |

## Items requiring author confirmation before repository deposition

| Item | Reason |
| --- | --- |
| Exact training/validation split used during hyperparameter selection | Needed for full reproducibility if not already fixed in final code |
| Final selected alpha if strengthened supervision was used in the locked model | Needed to avoid ambiguity between development variants and final model |
| Any probability calibration step | Needed if calibrated probabilities are reported rather than assignment probabilities |
| Final software environment | Needed for code deposition and reproducibility |
| Final model checkpoint availability | The locked local run retained assignment outputs, summary metadata and training history, but did not serialize a standalone `.pt` checkpoint; rerun with checkpoint serialization before public model release if executable assignment is required |

## Conservative manuscript interpretation

The model is best described as a fixed-label temporal assignment model for mapping incomplete laboratory trajectories to previously discovered recovery phenotypes. It supports external risk-ordering analyses but does not establish exact waveform equivalence, fully independent physiological effects or a validated treatment directive.
