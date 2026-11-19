# Results

Generated captions and every score behind the paper's tables and figures, under the paper's names.
The trained weights are on the Hugging Face Hub (see the repository README).

| Folder | Content |
|---|---|
| `seed42/<caption type>/<model>/` | `captions.json`, `clipscore.json` and `clipscore_per_image.json`, `siglip.json`, `reference_scores.json` (METEOR), `chair_per_image.json` |
| `seed42/geometry/<model>.json` | location, scale, shape and orientation measures on the 5,000 evaluation images |
| `seed1337/` | the same for the second seed (brief captions and geometry) |
| `tests_of_the_objective/` | CLIPScore and geometry of the variants of the location drive |
| `where_the_gap_comes_from/` | geometry with a random connector, after Stage 1 only, Stage 2 only, and both stages |
| `released_models_gap/` | geometry of LLaVA-1.5-7B, LLaVA-1.5-13B and bakLLaVA |
| `summaries/` | CHAIR summaries, inference cost, and the inference-time shift of the baseline |

Caption types: `brief` (MSCOCO val2017, 5,000 images, at most 64 new tokens), `long` (first 1,300
images, at most 256 new tokens), `nocaps` (300 NoCaps images).

Models: `baseline`, `constrained_baseline`, `Cloc`, `Cscale1500`, `Crank20`, `Corient`, `Clocorient`,
and Cloc trained longer (`Cloc_48.9k_items`, `Cloc_80k_items`); `LLaVA-1.5-7B` is the released
reference. In the geometry files, `G_mu` is the centroid distance, `trace_image` the total variance,
`eff_rank_image` the effective rank and `subspace_overlap_q["16"]` the subspace overlap.

These files were assembled by `scripts/release/build_data.py`; the weights by
`scripts/release/strip_weights.py`. Licences: see `LICENSE-NOTICE.md`.
