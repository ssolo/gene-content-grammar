# `scripts/` -- evaluation, analysis, and data-prep tooling

This directory holds everything that is *not* the core model/training package
(`ising_denoiser/`) or the SLURM batch wrappers (`slurm/`). The scripts fall
into a few families: data preparation, denoising-**spectra** evaluation,
**calibration** evaluation, cross-run aggregation, ancestral-genome
reconstruction/analysis, and diagnostics/profiling.

If you only want one thing: to characterise how well a checkpoint denoises and
how well-calibrated its probabilities are across the noise plane, use
**`sweep_spectra_calibration.py`** (driven by `slurm/run_spectra_sweep.sh`,
aggregated by `aggregate_spectra_sweep.py`).

---

## 1. Background: what we measure and how

The model is an iterative Ising-style denoiser over a fixed vocabulary of
~4789 COG gene families. A genome is a {-1, +1} presence/absence vector. We
corrupt a held-out genome with two independent noise channels and ask the model
to reconstruct the clean genome:

* **FN (false negative)** -- flip *present* genes to *absent* (`+1 -> -1`) with
  probability FN. This is the dominant channel for ancestral reconstruction
  (deep ancestors look gene-poor); the operating range for LBCA/LUCA is
  FN = 0.75-0.90.
* **FP (false positive)** -- flip *absent* genes to *present* (`-1 -> +1`) with
  probability FP. Models are trained at a small fixed FP (~0.01); robustness to
  larger FP is what the FP fine-tune targets.

Two complementary quality axes:

* **Spectra** = reconstruction *accuracy* vs noise. Primary metric **MCC**
  (Matthews correlation; robust to the heavy class imbalance -- only ~30% of
  COGs are present), plus F1/precision/recall and `fp_removed` (the fraction of
  injected false positives the model switches back off). Threshold: output
  sign (probability > 0.5). Implemented in `ising_denoiser/metrics.py`
  (`metrics`, `eval_spectra`, `eval_spectra_2d`).
* **Calibration** = probability *reliability*. Does a predicted 0.7 mean 70%
  present? Metrics: **ECE** / **MCE** (reliability-bin gaps), **Brier**, **NLL**.
  Implemented in `scripts/calibration_analysis.py` (equal-frequency bins,
  bootstrap CIs) and recomputed cheaply in `sweep_spectra_calibration.py`
  (fixed-width bins; see the binning note below).

### Deterministic noise (why spectra are comparable across checkpoints)

Noise is drawn with a fixed seed per cell, so the *same* genes flip for every
checkpoint and every GPU count:

```
seed = 42 + int(fn*1000) + 7919*int(fp*1000)      # eval_spectra_2d / sweep
```

(`calibration_sweep.py` uses a slightly different but equally deterministic
seed; it predates the 2D scan.) The noise is applied to the full sampled tensor
*before* DDP sharding, so results are independent of world size.

### Model-class auto-detection (one loader for every architecture)

`analyze_ancestral_node.py::build_model_for_ckpt` inspects state-dict keys and
the `model_*.cfg.json` sidecar to build the right class -- reused by
`sweep_spectra_calibration.py`:

| has `attn.*` | has `A` & `W` | class detected           | model                         |
|:------------:|:-------------:|--------------------------|-------------------------------|
| yes          | yes           | `higher_order`           | `HigherOrderDenoiser`         |
| yes          | no            | `nohidden_higher_order`  | `NoHiddenHigherOrderDenoiser` |
| no           | yes           | `hidden`                 | `ModuleConditionedDenoiser`   |
| no           | no            | `nohidden`               | `NoHiddenDenoiser`            |

`T` is read from the `skip_alpha`/`skip_gates` shape; `onsager` from the cfg
sidecar (falling back to `tied` for higher-order, `full`/bool otherwise -- which
is exactly correct for the cfg-less `gsd_results_higher_order_tied_T8` family).

---

## 2. Shared building blocks (`ising_denoiser/`)

| module        | what it provides                                                                 |
|---------------|----------------------------------------------------------------------------------|
| `data.py`     | `load_feathers(train, val, frac)` -> ({-1,+1} tensors, COG vocab). `frac=1.0` keeps file order. |
| `models.py`   | the four denoiser classes above (+ `SetTransformerDenoiser`).                    |
| `metrics.py`  | `metrics`, `eval_spectra`, `eval_spectra_2d`, `eval_spectra_multistep`, `FN_GRID`, `FP_GRID`. |
| `modules.py`  | `load_module_matrix` -> KEGG/COG module membership conditioning matrix.          |
| `training.py` | DDP primitives (`setup_dist`, `is_main`, `barrier`, `log`), checkpointing, `strip_compile_prefix`, T-extension. |
| `report.py`   | `eval_null`, `generate_report` (spectra report scaffolding).                     |

---

## 3. Evaluation -- denoising spectra (MCC / F1 vs noise)

| script                          | scope / GPUs        | grid                         | output                              |
|---------------------------------|---------------------|------------------------------|-------------------------------------|
| **`sweep_spectra_calibration.py`** | 1 ckpt, **DDP**  | fine 2D, FN 0.05 x FP {.01,.02,.05,.1} | parquet (spectra **and** calibration) |

## 4. Evaluation -- calibration (probability reliability)

| script                       | scope / GPUs | what it computes                                                            |
|------------------------------|--------------|-----------------------------------------------------------------------------|
| **`calibration_analysis.py`**| **DDP**      | headline ECE/MCE/Brier/NLL with **per-genome bootstrap CIs**, 7 reliability figures (Cal1-7), 3 LaTeX tables; per-COG / per-category / per-KEGG-module breakdowns. Supports `--val-domain`. |
| `calibration_sweep.py`       | **DDP**      | fine FN/FP **MCC/F1/prec/rec** sweep for `hidden` vs `nohidden` (NON-attention) models + publication plots. (Despite the name it reports accuracy, not ECE; cannot load attention checkpoints.) |

**Binning note.** `calibration_analysis.py` uses **equal-frequency** reliability
bins; `sweep_spectra_calibration.py` uses **fixed-width** bins (fast, O(N), so it
can score 70+ checkpoints). Their ECE numbers therefore differ slightly by
construction; Brier and NLL are binning-independent and match. Use
`calibration_analysis.py` for headline numbers, the sweep for broad exploration.

## 5. Aggregation (across splits / families)

| script                          | input                                            | output                                          |
|---------------------------------|--------------------------------------------------|-------------------------------------------------|
| **`aggregate_spectra_sweep.py`**| `spectra_sweep/*.parquet`                        | `data/spectra_calibration_all.parquet` + `_summary.csv` + `_SUMMARY.txt` |
| `aggregate_calibration_ft.py`   | `calibration_T20_ft_split*/predictions/*.parquet`| pre-FT vs mix-FT global/category summary + `SUMMARY.txt` |

## 6. Ancestral reconstruction & downstream analysis

| script                       | purpose                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `analyze_ancestral_node.py`  | ensemble-denoise an ancestral node; writes report + LLM prompt. Hosts the reusable `build_model_for_ckpt`. |
| `extract_ancestral_node.py`  | convert Davin et al. ancestral-reconstruction tables to per-COG probs.  |
| `extract_laca_node.py`       | convert a long-format archaeal per-species table to a LACA node vector. |
| `denoise_genome.py`          | run a trained denoiser on a single partial genome.                      |
| `plot_laca_ho.py`            | HO_tied LACA denoising plots.                                           |
| `module_completeness.py`     | KEGG-module / COG-pathway completeness audit of a (denoised) gene set.  |

## 7. Data preparation

| script                        | purpose                                                              |
|-------------------------------|----------------------------------------------------------------------|
| `download_validation_data.py` | download STRING PPI data + COG mappings.                             |
| `build_feathers.py`           | build train/val feathers with whole-clade holdout (the original 10-fold). |
| `build_archaeal_feathers.py`  | build the separate archaea-only 10-fold (COG_arc / COG_mix).        |
| `build_bacterial_feathers.py` | build the separate bacteria-only 10-fold (COG_bac).                 |
| `build_module_matrix_kegg.py` | build `data/module_matrix_kegg.pt` (COG -> KEGG module membership). |

Shell helpers (`launch_ho_*.sh`) are thin convenience wrappers; the canonical
batch entry points live in `slurm/`.

---

## 8. Reproducing the broad spectra + calibration sweep

This produces the committed plotting dataset across all higher-order families
(pre arch-FT T8/T12/T16/T20, T20 mix-FT, T20 arc-FT, hidden HO T8):

```bash
# 1. on the cluster: run the resume-safe 8-GPU sweep (one parquet per
#    family x split; skips any that already exist; auto-resubmits if the
#    12h wall is hit).
sbatch slurm/run_spectra_sweep.sh
#    archaea-only variant:
sbatch --export=ALL,VAL_DOMAIN=d__Archaea,OUTROOT=spectra_sweep_arc slurm/run_spectra_sweep.sh

# 2. pull results back (small parquets only):
rsync -av <cluster>:$PROJECT_ROOT/ ./spectra_sweep/

# 3. aggregate into the committed dataset:
python3 scripts/aggregate_spectra_sweep.py \
    --results-glob 'spectra_sweep/*.parquet' \
    --out data/spectra_calibration_all.parquet

# 4. (re)generate every figure into analysis/figures/:
python3 scripts/plot_results.py

# 5. compile the write-up to analysis/report.pdf.  build_results.sh prefers a
#    local TeX install (MacTeX in /Library/TeX/texbin, often missing from a
#    non-login PATH) and runs pdflatex twice; if no local TeX is found it falls
#    back to compiling on the GPU cluster and rsyncing the PDF back.  The PDF is a
#    regenerable, gitignored artifact.  Works on committed or WIP drafts.
scripts/build_results.sh
```

Each row of `data/spectra_calibration_all.parquet` is one
`(family, split, fn, fp, step)` cell -- `step` is the model's internal
annealing iteration (`step=1..T`, final step `T` is byte-identical to the
single-pass output; the model-free null baseline is `step=0`) -- carrying
both spectra (`MCC F1 prec rec fp_removed base_rate`) and calibration
(`ece mce brier nll`), everything needed to build the plots (and the per-step
trajectory figures) without re-running any inference.
