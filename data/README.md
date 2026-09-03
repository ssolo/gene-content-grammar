# Reconstruction & figure-source data tables

Per-COG reconstruction tables and the figure-source data derived from them. Every
table is a TSV over the same **N = 4,789 COG families** (rows keyed by `COG_ID`,
`COG0001 …`). The training/validation feathers and the module matrix are documented
in the top-level `README.md`; this file documents the reconstruction outputs.

## Columns shared by all tables

| column | meaning |
| --- | --- |
| `COG_ID` | COG family id, the row key (same 4,789 set in every table) |
| `category` | single-letter COG functional category (C, E, H, P, …) |
| `name` | COG description |

`category`/`name` are joined from the COG annotation carried in
`lbca_cog_lists.tsv` / `laca_cog_lists.tsv`.

## Per-condition column convention

Each reconstruction (or noise condition) contributes a group of columns sharing a
prefix `<x>`:

| suffix | meaning |
| --- | --- |
| `<x>_input` | the input the model was given (reconciliation / copy-number call or corrupted observation), in `[0,1]` |
| `<x>_out_post` or `<x>_post` | denoised posterior `P(present)` — ensemble mean over the ten cross-validation models, in `[0,1]` |
| `<x>_sd` | ensemble s.d. of that posterior |
| `<x>_present` | binary present call — `1` iff posterior `> 0.5` (the threshold used throughout the paper) |
| `<x>_state` | (ground-truth tables only) correctness label — see `ecoli_cog_lists.tsv` |

---

## Ancestral reconstructions (Figs 6–7)

Built by `scripts/build_all_reconstructions.py` (per-COG lists) and
`scripts/build_root_consensus.py` (consensus), from the denoiser ensemble applied to
each upstream reconstruction input.

| file | rows × cols | conditions `<x>` (each ×{`_input`,`_out_post`,`_present`}) | figure |
| --- | --- | --- | --- |
| `lbca_cog_lists.tsv` | 4789 × 12 | `sparse`, `GLD_min1`, `GLD_min4` | Fig 6 (LBCA) |
| `laca_cog_lists.tsv` | 4789 × 18 | `combined`, `euryroot_ml`, `euryroot_unif`, `gld_min1`, `gld_min4` | Fig 7 (LACA) |

`{lbca,laca}_consensus.tsv` — ensemble consensus across those variants:
`mean_post, median_post, sd_post, n_present, n_recon, consensus_present, vote`, then the
per-variant `<x>_post`.

## E. coli ground-truth recovery (Fig 3)

`ecoli_cog_lists.tsv` — 4789 × 16 — built by **`scripts/build_ecoli_cog_lists.py`**.
Source: the exact Figure-3 data
`analysis/leakage_audit/scratch_ecoli_split5/recover_ecoli_marginal_hq_split5_typical_fn468.tsv`
(production denoiser **bac-FT-fp-marginal-HQ**, split 5, E. coli held out at both
training stages, `fp=0.01`).

| column | meaning |
| --- | --- |
| `truth` | E. coli ground-truth presence (0/1); 2,178 genes present |
| `fnXX_input` / `_post` / `_present` | input / denoised posterior / call at false-negative rate `fn ∈ {0.4, 0.6, 0.8}` |
| `fnXX_state` | `kept` (true, in input) · `recovered` (true, deleted by noise, restored) · `gap` (true, not recovered = false negative) · `FP` (false positive) · `absent` (true negative) |

Reproduces Fig 3: at `fn=0.8`, recall 58.6 %, precision 88.6 %. Companion
`ecoli_cog_lists.meta.tsv` records the model, thresholds, and per-`fn` recall/precision.

## LACA Jaccard convergence (Fig 8)

Built by **`scripts/build_laca_figure8_data.py`** from the five pred files that
`scripts/plot_laca_jaccard_heatmap.py` loads (`LACA_merged_pred.tsv`,
`LACA_euryroot_pred.tsv`, `LACA_euryroot_uniform_pred.tsv`, `LACA_gld_min1_pred.tsv`,
`LACA_gld_min4_pred.tsv`; source columns `input_prob`, `mean_actual`, `sd_actual`).

`laca_figure8_cog_lists.tsv` — 4789 × 23. Reconstruction key ↔ Figure-8 label:

| key `<x>` | Figure-8 label | pred file |
| --- | --- | --- |
| `combined_root` | combined root | `LACA_merged_pred.tsv` |
| `euryarch_ml` | Euryarchaeota (ML) | `LACA_euryroot_pred.tsv` |
| `euryarch_unif` | Euryarchaeota (uniform) | `LACA_euryroot_uniform_pred.tsv` |
| `copynum_min1` | copy-number (min-1) | `LACA_gld_min1_pred.tsv` |
| `copynum_min4` | copy-number (min-4) | `LACA_gld_min4_pred.tsv` |

each ×{`_input` (=`input_prob`), `_post` (=`mean_actual`), `_sd` (=`sd_actual`),
`_present` (post > 0.5)}.

`laca_figure8_jaccard.tsv` — the numbers actually plotted (tidy form):
`panel` (`before_input` / `after_denoised`), `recon_i`, `recon_j`, `jaccard`, plus a
`MEAN_OFFDIAG` row per panel. Reproduces the caption: mean off-diagonal Jaccard
**0.23 → 0.71**.

---

To rebuild any table: `python3 scripts/<builder>.py`. All present calls use posterior `> 0.5`.

## Third-party data

`interactome/goodall_st1.xlsx` — Table S1 of Goodall et al. (2018), *The Essential
Genome of Escherichia coli K-12*, mBio 9:e02096-17, doi:10.1128/mBio.02096-17.
Redistributed unmodified under the article's CC BY 4.0 licence, and used for the
experimental essentiality calls in `scripts/interactome/essentiality_recon.py`.
It is **not** covered by this repository's CC BY-NC licence; see LICENSE.
