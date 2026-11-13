# Single-Axis Attribution of the Modality Gap in Multimodal Language Models

> Master's thesis · Politecnico di Torino, Data Science and Engineering, a.y. 2025–2026
> Hassine El Ghazel · supervisors prof. Giuseppe Rizzo, Dr. Federico D'Asaro, Dr. Luca Catalano

---

## What this asks

Corrections for the modality gap move several geometric components at once, whether by a fixed map
over frozen embeddings or by a separately trained alignment, and report one aggregate score. Credit
is never divided between the components moved, so which of them a decoder is sensitive to remains
unknown.

This work asks that question inside a **connector-based multimodal language model**, where the
generated caption is the only read-out. In this model, instruction tuning on its own moves the image
centroid **38 %** further from the text centroid, from 177.67 to 245.46, because nothing in the
standard recipe prevents it.

The gap is decomposed into **four measurable axes**. Each is driven in turn while the other three are
**pinned** at their pre-intervention values, and the pins are verified from the trained model rather
than assumed.

---

## Architecture

Fixed across every condition. The connector is the only component whose output geometry varies.

<p align="center">
  <img src="docs/img/architecture.png" alt="Image path through frozen CLIP ViT-L/14 and a trainable MLP connector, text path through the frozen embedding table, both feeding a frozen 4-bit LLaMA-2-7B with LoRA adapters" width="560">
</p>

| component | detail |
|:---|:---|
| Vision encoder | CLIP ViT-L/14 at 224 px, frozen, 257 visual tokens |
| Connector | `mlp2x_gelu`, 1024 → 4096 → 4096, 21.0 M parameters |
| Decoder | LLaMA-2-7B, 4-bit NF4 with double quantisation, bfloat16 compute |
| Adapters | LoRA r = 8, α = 16, dropout 0.05, on `q_proj` / `v_proj` / `o_proj` |
| Measurement point | connector output, 4096-d, mean over the 257 projected tokens |

---

## Training

Two stages, following the LLaVA recipe.

<p align="center">
  <img src="docs/img/training_stages.png" alt="Left: Stage 1, contrastive connector alignment against the frozen embedding table. Right: Stage 2, instruction tuning with a loss on response tokens only" width="820">
</p>

<p align="center"><em>Left: Stage 1, connector alignment. Right: Stage 2, instruction tuning.</em></p>

**Stage 1 · connector alignment.** Only the connector moves. A symmetric InfoNCE objective aligns the
connector's `[CLS]` output with the mean-pooled lookups of the decoder's frozen embedding table: CLIP
with the text tower replaced by a fixed one. Data: Bunny-v1.1.

**Stage 2 · instruction tuning.** The connector is refined together with the LoRA adapters under an
autoregressive cross-entropy loss on response tokens only, so on its own this stage gives the
connector no direct signal about the geometry it produces. The geometry terms enter here. The
proposed configuration, **Cloc**, is

$$\mathcal{L} = 0.9\,\mathcal{L}_{\mathrm{AR}} + 0.1\,\mathcal{L}_{\mathrm{dist}}
              + 1.0\,\mathcal{L}_{\mathrm{scale}} + 1.0\,\mathcal{L}_{\mathrm{rank}}$$

The driven term is paid for out of the language budget through a convex combination; the two pins are
additive guards that hold scale and shape at their post-Stage-1 setpoints. Moving a pin's setpoint
instead of freezing it is what turns a pin into a drive.

| | Stage 1 | Stage 2 |
|:---|:---|:---|
| Objective | symmetric InfoNCE | AR cross-entropy (+ geometry terms) |
| Trainable | connector | connector and LoRA |
| Data | Bunny-v1.1 | LLaVA-Instruct-150K, ≈ 14,400 items |
| Learning rate | 5e-4, cosine, 3 % warm-up | 2e-4, cosine, 3 % warm-up |
| Batch | 64 | 4 × 8 accumulation (effective 32) |

Every condition runs **450 optimizer steps**, about **9.1 % of one epoch**, on a **single RTX 2080 Ti
(11 GB)**. bfloat16, seed 42.

---

## The four axes

Let $\bar x, \bar y$ be the cloud means and $\Sigma_X$ the image covariance, with eigenvalues
$\lambda_1 \ge \dots \ge \lambda_d$ and $d = 4096$. Everything is measured **at the connector
output**, because that is the input the decoder actually receives.

| axis | definition | plain meaning |
|:---|:---|:---|
| **Location** | $G_\mu = \lVert \bar x - \bar y \rVert$, $\;\widehat G_\mu = G_\mu / \sqrt{\mathrm{tr}\Sigma_X}$ | distance between the two centres |
| **Scale** | $s = \mathrm{tr}\Sigma_X = \sum_j \lambda_j$ | total variance of the image cloud |
| **Shape** | $r = (\mathrm{tr}\Sigma_X)^2 / \mathrm{tr}(\Sigma_X^2) \in [1,d]$ | how many directions the variance occupies |
| **Orientation** | $O_q = \frac{1}{q}\lVert U_X^{(q)\top} U_Y^{(q)}\rVert_F^2 = \frac{1}{q}\mathrm{tr}(P_X P_Y)$ | overlap of the principal subspaces |

The diagnostic computes twenty scalars per checkpoint; these four survive three constraints:
non-degeneracy, non-redundancy, and controllability by a differentiable penalty on a single batch.
Orientation is read as a multiple of chance $q/d$; results use $q = 16$, where chance is
$\approx 3.9 \times 10^{-3}$.

<p align="center">
  <img src="docs/img/interventions.png" alt="Five panels: identity, location (mean-shift), orientation (rotate), scale (shrink) and shape (anisotropy), each showing the image cloud before and after against a fixed text cloud" width="900">
</p>

<p align="center"><em>The single-axis interventions. The text cloud is fixed because it comes from the frozen embedding table.</em></p>

---

## Results

### Isolation is measured, not assumed

Geometry each condition reached, n = 5,000. Bold marks the targeted axis.

| condition | axis released | $G_\mu$ | $\mathrm{tr}\Sigma_X$ | $r$ | $O_{16}$ |
|:---|:---|---:|---:|---:|---:|
| all-pinned | none (baseline) | 177.86 | 4252.7 | 36.99 | 0.0515 |
| Cloc | location | **30.97** | 4789.6 | 37.32 | 0.0519 |
| Cscale1500 | scale | 177.74 | **1487.6** | 38.67 | 0.0435 |
| Crank15 | shape | 177.81 | 4311.6 | **20.21** | 0.0517 |
| Corient | orientation | 177.61 | 4578.8 | 35.95 | **0.1114** |
| Clocorient | location + orientation | **11.72** | 4804.1 | 36.32 | **0.2993** |

Each intervention moved its own axis by 45 % to 116 % while every off-target axis moved by at most
16 %. The largest drift, Cloc's +12.6 % in total variance, bounds to roughly 12 % of its gain by
extrapolating from the scale condition.

### Downstream effect

Against all-pinned. Brief captioning, MSCOCO `val2017`, n = 5,000; CLIPScore differences with 95 %
paired bootstrap intervals over images.

| condition | CLIPScore | Δ vs all-pinned [95 % CI] | verdict |
|:---|---:|:---|:---|
| all-pinned | 0.5799 | | reference |
| **Cloc** | **0.6325** | **+0.0526** [+0.0492, +0.0559] | **the lever** |
| Cscale1500 | 0.5466 | −0.0333 [−0.0365, −0.0299] | the mirror |
| Crank15 | 0.5769 | −0.0030 [−0.0055, −0.0007] | no effect that replicates ¹ |
| Corient | 0.5950 | +0.0151 [+0.0124, +0.0178] | weak, about a third of location |
| Clocorient | 0.6421 | +0.0622 [+0.0589, +0.0656] | confounded ² |

Detailed description (`dd256`), n = 1,300:

| condition | CLIPScore ↑ | CHAIR<sub>s</sub> ↓ | CHAIR<sub>i</sub> ↓ | recall ↑ | objects / caption |
|:---|---:|---:|---:|---:|---:|
| all-pinned | 0.5703 | 0.7885 | 0.4396 | 0.4529 | 8.69 |
| **Cloc** | **0.6187** | **0.7369** | **0.3698** | **0.5116** | **8.98** |
| Cscale1500 | 0.5255 | 0.8900 | 0.5941 | 0.3558 | 8.15 |
| Crank15 | 0.5673 | 0.7923 | 0.4540 | 0.4595 | 9.17 |
| Corient | 0.5787 | 0.7838 | 0.4481 | 0.4524 | 8.31 |

**Location is a lever.** Closing the centroid distance is the only single-axis intervention that
improves every measure. It lowers both hallucination rates while the model names *more* objects, so
it is not a say-less trade. **Scale is the mirror:** compressing total variance moves every brief-set
measure the other way. **Orientation**, the axis a contrastive objective controls most directly,
returns about a third of the location effect.

A perturbation that helps, one that hurts and one that does neither cannot all be generic disturbance
of the decoder's input, so what matters is which axis moved. CLIPScore on both sets, SPICE and METEOR
order the conditions identically: Clocorient > Cloc > Corient > all-pinned > Crank15 > Cscale1500.

¹ Halving the effective rank shifts brief-set CLIPScore by −0.0030, 6 % of the location effect. The
interval excludes zero but does not survive correction for the fifteen comparisons, and the effect
is not resolved on `dd256`.
² Clocorient's centroid also closes further, from 30.97 to 11.72, so its extra gain cannot be
credited to orientation.

### A qualitative example

<p align="center">
  <img src="docs/img/qualitative.png" alt="A photograph of grazing zebras above six generated descriptions: the four conditions that leave the location gap open call the animals horses or describe a living room; the two location-driven conditions name the zebras" width="560">
</p>

Every condition that leaves the location gap open calls the zebras horses, and compressing the scale
abandons the scene for a living room. Both conditions driven by the location term name the zebras.

### Distance to a fully resourced reference

`dd256`, n = 1,300. LLaVA-1.5-7B is evaluated for inference only on the same protocol.

| | steps | items | CLIPScore ↑ | CHAIR<sub>s</sub> ↓ | CHAIR<sub>i</sub> ↓ | recall ↑ | words |
|:---|---:|---:|---:|---:|---:|---:|---:|
| all-pinned | 450 | 14,400 | 0.5703 | 0.7885 | 0.4396 | 0.4529 | 94.9 |
| Cloc | 450 | 14,400 | 0.6187 | 0.7369 | 0.3698 | 0.5116 | 100.1 |
| Clocorient | 450 | 14,400 | 0.6312 | 0.7638 | 0.3656 | 0.5336 | 96.0 |
| Cloc_long | 1,529 | 48,928 | 0.6886 | 0.7869 | 0.3314 | 0.6393 | 101.6 |
| Cloc_80k | 2,500 | 80,000 | 0.7033 | 0.7762 | 0.3133 | 0.6675 | 103.9 |
| LLaVA-1.5-7B (reference) | n/a | ≈ 1.2 M | 0.7910 | 0.5069 | 0.1518 | 0.7738 | 82.4 |

The best 450-step condition covers about a quarter of the distance from all-pinned to the reference
on every measure: 28 % on CLIPScore, 26 % on CHAIR<sub>i</sub>, 25 % on recall. The reference differs
in every advantageous dimension (336 px and 576 visual tokens, an instruction-tuned decoder, ~1.2 M
samples); it is a reference, not a competitor.

Geometry gives the sharper comparison. On the same 1,300 pairs the reference's normalised location
gap is $\widehat G_\mu = 3.250$, wider than all-pinned's 2.727 and far from Cloc's 0.448: an order of
magnitude more training data does not close this axis on its own.

---

## Evaluation

| regime | data | metrics |
|:---|:---|:---|
| Brief captioning | MSCOCO `val2017`, 5,000 images, 64 new tokens | CLIPScore, SPICE, METEOR |
| Detailed description (`dd256`) | first 1,300 of `val2017`, 256 new tokens | CLIPScore, CHAIR<sub>s</sub>, CHAIR<sub>i</sub>, object recall |

CLIPScore is computed with ViT-B/32, not the model's own ViT-L/14, so the judge is not the tower
that produced the features. SPICE and METEOR compare against human references, and CHAIR counts
objects named but absent from the COCO annotation. n-gram metrics are not used on `dd256`, because
they need short reference captions.

---

## What this budget leaves open

- 450 optimizer steps on a single 11 GB GPU. The models are instruments for an attribution, not
  competitive captioners, and every claim is a contrast between conditions sharing that budget
  exactly. An effect at this budget shows an axis is live; a null is the weaker claim.
- One training seed per condition. The intervals above describe variation across evaluation images,
  not across training runs.
- Only the location drive was carried past 450 steps, so the longer rungs have no matched
  all-pinned control at 1,529 or 2,500 steps.
- One architecture and one vision tower, and captioning only.

---

## Repository layout

```
configs/      YAML for encoders, projector, LLM, both training stages and the eval regimes
src/
  encoders/     frozen CLIP wrapper
  models/       connector, VLM splice, checkpoint I/O
  training/     stage 1 and stage 2 loops, geometry objectives
  diagnostics/  projected-embedding extraction, the 20-statistic suite, plots
  evaluation/   captioning and scoring helpers
  data/         COCO, Bunny, LLaVA-Instruct, DOCCI loaders
scripts/      numbered pipeline steps (03 gap, 05/06 train, 08 caption, 15 CLIPScore, 19 CHAIR, ...)
              plus the Slurm submission scripts for each condition
demo/         live side-by-side demo (see README_DEMO.md)
outputs/      checkpoints, predictions, metrics, logs
tests/        unit tests
```

## Reproducing

```bash
bash scripts/00_setup_env.sh          # Python 3.10, CUDA 12.1; uv preferred
bash scripts/01_download_data.sh

python scripts/05_train_stage1.py --config configs/training_stage1.yaml
python scripts/06_train_stage2.py --config configs/training_stage2.yaml

python scripts/07_extract_projected.py --condition <tag>
python scripts/03_compute_gap.py       --condition <tag>
python scripts/08_run_captioning.py    --condition <tag> --vlm-checkpoint <ckpt>
python scripts/15_clipscore.py         --condition <tag>
python scripts/19_chair.py             --conditions <tag> --baseline C3pinr_dd256
```

The single-axis conditions are trained by `scripts/17_train_c5_distance.py` (Cloc) and
`scripts/18_train_c6_orientation_pinned.py` (all-pinned, Cscale1500, Crank15, Corient, Clocorient);
the exact flags for each are in `scripts/submit_pooled.sbatch`. `15_clipscore.py` also writes
per-image scores, which the paired bootstrap above uses. `java` is required on `PATH` for METEOR and
SPICE.

---

## Using this work

**Code** in this repository is licensed under the **GNU AGPL-3.0-or-later** (see `LICENSE`).
You may use, study, modify and redistribute it, including for research, provided derivative
works are released under the same licence. If you run a modified version as a network
service, the AGPL requires you to offer its source to users of that service.

**The thesis and paper text, their figures and the generated captions** are not covered by that
licence. All rights in them are reserved by the author.

Vendored third-party files retain their original licences; see `THIRD_PARTY_LICENSES.md`.

If you use this code or results derived from it, please cite the thesis. `CITATION.cff`
carries the machine-readable record, and GitHub's "Cite this repository" button reads it.

Copyright (c) 2026 Hassine El Ghazel.
