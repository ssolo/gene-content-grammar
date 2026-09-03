# Ising Denoiser — Ancestral Genome Reconstruction

Probabilistic reconstruction of ancestral gene content from noisy phylogenetic reconciliation data, using an iterative mean-field Ising model with learned pairwise couplings, Onsager (TAP) correction, hidden state, and module-completeness conditioning.

## Overview

Gene tree–species tree reconciliation methods (ALE, GeneRax, ALeRax) infer which genes were present at ancestral nodes, but suffer from substantial false-negative rates (FN = 0.5–0.9 at deep nodes like LBCA/LUCA). This tool denoises reconciliation output by learning the pairwise co-occurrence structure of genes across extant prokaryotic genomes and using it to reconstruct missing genes.

## Environment

Full install details are in [INSTALL.md](INSTALL.md). For the common case,
create or refresh the project virtualenv from a fresh checkout with:

```bash
./setup_env.sh
source .venv/bin/activate
```

The setup script prefers `python3.11`, installs `requirements.txt`, and verifies
that the modules used by the package and scripts import correctly. Large COG
feather inputs and model checkpoints are supplied separately because of their
size.

The trained models and training data are archived on **Zenodo** under the
concept DOI [`10.5281/zenodo.20526264`](https://doi.org/10.5281/zenodo.20526264),
which always resolves to the latest version. The record holds the production
model checkpoints (10 cross-validation splits each), the training and validation
feathers, and the COG vocabulary.

## Focal model

The focal architecture is a `HigherOrderDenoiser` — a `ModuleConditionedDenoiser` (pairwise J, hidden state z ∈ ℝ^H, adaptive temperature, module-completeness conditioning, Onsager TAP) augmented with an attention-parameterised higher-order correction Δ_t(x, z). (The **production** models drop the hidden state z: the `NoHiddenHigherOrderDenoiser` matches this at lower cost and is what `TRAINING.md` trains and `scripts/reconstruct.py` serves; the with-hidden form shown here is the architectural reference.) Given a noisy genome x₀ ∈ {±1}^N over N=4789 COG families, it iteratively refines its reconstruction over T iterations:

    coupling = x^t J̃ + z^t Aᵀ − x^t (1 − x^t²) @ J²     (TAP/Onsager correction)

    x^{t+1} = tanh( (h + (1 + (1+γ_field)·φ_t(x₀)) · coupling
                       + (1+γ_skip)·σ_t(x₀) · x₀
                       + Δ_t(x^t, z^t)) / τ_t )

    z^{t+1} = tanh( x^{t+1} Uᵀ + z^t W̃ − z^t (1 − z^t²) @ W²  )

where:

- **J ∈ ℝ^{N×N}** — symmetric zero-diagonal pairwise coupling matrix encoding gene co-occurrence
- **h ∈ ℝ^N** — per-gene bias (external field)
- **z ∈ ℝ^H** — hidden state (H=1000) capturing higher-order interactions via A, U, W
- **σ_t(x₀)** — skip gate: `α_t + w_t ⊙ x₀ + x₀ V_t V_tᵀ` (scalar + diagonal + low-rank)
- **φ_t(x₀)** — field gate: `u_t ⊙ x₀ + d_t + x₀ P_t P_tᵀ` (diagonal + low-rank)
- **γ_skip, γ_field** — per-timestep conditioning scalars from MLP(f_obs), where f_obs = Mᵀ((x₀+1)/2)/|M| is the observable module-completeness profile over 419 functional modules
- **τ_t(ρ)** — adaptive temperature: `softplus(a_t + b_t · ρ)`, where ρ is input gene density
- **Onsager/TAP** — the `−x(1−x²)@J²` term is the Thouless-Anderson-Palmer reaction-term correction, which removes the self-reinforcement bias of naive mean-field by subtracting the contribution of each spin to its own local field
- **Δ_t(x, z)** — higher-order correction: an attention-parameterised additive field (per-gene embeddings → `MemoryEfficientEncoder` → linear head) reading x^t, z^t, and a per-timestep embedding. Δ enters as a parallel residual to the pre-tanh field, **not** modulated by the field gate. The output head is zero-init so Δ ≡ 0 at the start of HO training and the model reduces exactly to its pairwise parent; Δ is then unfrozen via the HO1 → HO2 → HO3 curriculum (see [Higher-order denoiser](#higher-order-denoiser-train_higher_orderpy) below). Implemented as `HigherOrderModule` (`ising_denoiser/models.py`)

### Onsager modes

The `--onsager` flag controls the TAP correction:

| Mode | Correction | Use case |
|------|-----------|----------|
| `none` | No TAP correction | Baseline (naive mean-field) |
| `within` | J² on visible, W² on hidden | Within-block only |
| `full` | J², W², + cross-block A² and U² (independent A, U) | Asymmetric iteration |
| `tied` | J², W², + cross-block A² (U = Aᵀ tied) | Single-matrix EBM, proper block-TAP ELBO |

In `tied` mode, the hidden-to-visible coupling A and the visible-to-hidden coupling U are a single matrix (U = Aᵀ), so both cross-block Onsager terms are derived from A² alone. This corresponds to a proper energy-based model

    E(x, z) = −hᵀx − ½ xᵀJx − ½ zᵀWz − zᵀAx

with a well-defined Boltzmann distribution and a block-TAP ELBO. The iteration is exact coordinate ascent on a single scalar, so convergence is monotone.

### Ablations

Two orthogonal ablations dissect what each piece of the focal model contributes. Together with the focal `HigherOrderDenoiser`, they form a 2 × 2 matrix on {hidden state z, attention Δ}:

| Hidden state | No Δ | + Δ |
|---|---|---|
| **no z** | `NoHiddenDenoiser` (`--no-hidden`) | **`NoHiddenHigherOrderDenoiser`** |
| **+ z** | `ModuleConditionedDenoiser` (pairwise focal) | `HigherOrderDenoiser` (focal) |

- **`NoHiddenDenoiser`** — removes the hidden state entirely (no A, U, W; no z iteration; no module-completeness auxiliary loss). Coupling collapses to `x^t J̃` (+ Onsager). Isolates the contribution of the hidden layer to reconstruction. Trained with the `--no-hidden` flag in `train_denovo.py`.

- **`NoHiddenHigherOrderDenoiser`** — visible-only attention Δ on top of `NoHiddenDenoiser`. Removes z, *but keeps* the higher-order correction Δ_t(x, t) (attention now reads only x and the per-iteration t embedding, since there is no z to condition on). Module conditioning still feeds γ_skip / γ_field to the parent's gates; the module-completeness auxiliary loss is auto-skipped (z=None). Trained via `train_higher_order_nohidden.py` from a `NoHiddenDenoiser` checkpoint, using the same HO1/HO2/HO3 curriculum as the hidden variant.

The two ablations together answer two distinct questions:

1. **Is the hidden state z necessary?** (compare `NoHiddenDenoiser` vs `ModuleConditionedDenoiser`, and `NoHiddenHigherOrderDenoiser` vs `HigherOrderDenoiser`).
2. **Does Δ help over and above z?** (compare `NoHiddenHigherOrderDenoiser` vs `NoHiddenDenoiser`, and `HigherOrderDenoiser` vs `ModuleConditionedDenoiser`).

If Δ provides comparable lift in both rows of the matrix, the higher-order structure it captures is largely orthogonal to what z absorbs. If Δ's lift collapses in the +z row, then z was already capturing most of the higher-order signal and Δ is redundant.

## Repository structure

    ising_denoiser/
    ├── ising_denoiser/              # Core package
    │   ├── models.py                # ModuleConditionedDenoiser, NoHiddenDenoiser, HigherOrderDenoiser, SetTransformerDenoiser
    │   ├── modules.py               # ModuleCompletenessPredictor, module_aux_loss, load_module_matrix
    │   ├── data.py                  # ReconciliationNoiseDataset, K-replicate augmentation
    │   ├── metrics.py               # MCC/F1 metrics, FN-spectrum eval, comparison printing
    │   ├── training.py              # DDP, loss, scheduler, checkpointing, load_and_extend
    │   └── report.py                # Publication figures + LaTeX report generation
    ├── train_denovo.py              # Primary training: 8-stage tempered ELBO curriculum (T8->T20 chain)
    ├── train_higher_order_nohidden.py # Attention-Delta on a NoHiddenDenoiser checkpoint (HO1/HO2/HO3); production
    ├── train_add_hidden.py          # HO + explicit hidden units (the HO+hidden control)
    ├── scripts/                     # Utility, build, analysis, and orchestration scripts
    │   ├── reconstruct.py           # Unified CLI: reconstruct extant / ancestral genomes
    │   ├── reconstruct_extant.py    # Corrupt a held-out genome and denoise vs truth
    │   ├── analyze_ancestral_node.py    # Ensemble denoise of an ancestral node (LBCA/LACA)
    │   ├── extract_ancestral_node.py    # Davin-et-al. table -> per-COG presence probs
    │   ├── extract_laca_node.py     # Long-format archaeal table -> LACA node vector
    │   ├── denoise_genome.py        # Run a trained denoiser on a partial input
    │   ├── sweep_spectra_calibration.py # DDP spectra+calibration over the FN/FP plane
    │   ├── calibration_analysis.py  # Multi-GPU calibration analysis (DDP)
    │   ├── calibration_sweep.py     # Batched calibration over FN x FP grid
    │   ├── aggregate_spectra_sweep.py   # Collate sweep parquets -> committed dataset
    │   ├── aggregate_calibration_ft.py  # Pre-FT vs FT calibration summary
    │   ├── module_completeness.py   # KEGG-module completeness audit of a gene set
    │   ├── estimate_apparent_noise.py   # Invert the noise-meter for apparent fN/fP
    │   ├── plot_results.py          # Regenerate spectra/calibration figures
    │   ├── plot_recon_figs.py       # Regenerate the per-genome recovery figures
    │   ├── plot_fp_varying.py       # FP-grid spectra figure
    │   ├── plot_laca_ho.py          # LACA denoising plots
    │   ├── interactome/             # ProteomeLM-style STRING-PPI benchmark of J
    │   ├── launch_ho_*.sh           # Sequential queue launchers (chain, specialists, fp)
    │   ├── download_validation_data.py  # STRING PPI + COG mappings
    │   ├── build_feathers.py        # Build COG feathers (whole-phylum 10-fold)
    │   ├── build_bacterial_feathers.py  # Bacteria-only 10-fold (COG_bac)
    │   ├── build_archaeal_feathers.py   # Archaea-only 10-fold (COG_arc / COG_mix)
    │   └── build_module_matrix_kegg.py  # data/module_matrix_kegg.pt (COG -> KEGG module)
    ├── slurm/                       # SLURM training launchers (generic)
    │   ├── run_denovo_elbo.sh       # De novo ELBO training (with hidden)
    │   ├── run_nohidden_denovo_elbo.sh  # De novo ELBO training (no-hidden; the T8->T20 chain)
    │   ├── run_higher_order_nohidden.sh # Attention-Delta on a NoHiddenDenoiser checkpoint (HO)
    │   ├── run_higher_order_nohidden_T20_{mix,bac,arc}_finetune.sh # specialist fine-tunes
    │   ├── run_ho_addhidden_T20.sh  # HO+hidden control fine-tune
    │   ├── run_spectra_sweep.sh     # resume-safe 8-GPU spectra/calibration sweep
    │   ├── run_calibration.sh       # Multi-GPU calibration analysis
    │   ├── run_calibration_sweep.sh # Batched calibration sweep
    │   ├── run_calibration_T20_*.sh # specialist calibration eval
    │   ├── bac_fp_spectra_array.sbatch  # FP-grid spectra (8-way array)
    │   └── fp_compare.sbatch        # FP-tolerant vs default comparison
    ├── data/                        # Input data (feathers gitignored; matrices + tables tracked)
    └── outputs/                     # Example analysis outputs (e.g. outputs/node2012/)

## Data

| File | Description |
|------|-------------|
| data/COG_train{1..10}_phylum.feather | ~90K genomes × 4,789 COGs (10 cross-validation splits) |
| data/COG_val{1..10}_phylum.feather | ~23K genomes × 4,789 COGs (validation splits) |
| data/module_matrix_kegg.pt | Module membership matrix (4789 × 419) |
| data/*_cog_lists.tsv, *_consensus.tsv, laca_figure8_* | Per-COG reconstruction & figure-source tables (LBCA/LACA/E. coli) — columns and sources documented in [data/README.md](data/README.md) |

## Training

### De novo training (train_denovo.py) — primary entry point

Trains a `ModuleConditionedDenoiser` (or `NoHiddenDenoiser` with `--no-hidden`) from scratch using an 8-stage curriculum with ELBO loss and Onsager TAP correction.

#### Quick start (SLURM)

    # Full model with Onsager, T=12, all 10 splits:
    for SPLIT in {1..10}; do
      sbatch --export=ALL,EXTRA="--train-feather data/COG_train${SPLIT}_phylum.feather \
        --val-feather data/COG_val${SPLIT}_phylum.feather --onsager full \
        --batch-per-gpu 6000 \
        --outdir gsd_results_denovo_elbo_onsager_T12_split${SPLIT}" \
        slurm/run_denovo_elbo.sh 12
    done

    # No-hidden ablation:
    for SPLIT in {1..10}; do
      sbatch --export=ALL,EXTRA="--train-feather data/COG_train${SPLIT}_phylum.feather \
        --val-feather data/COG_val${SPLIT}_phylum.feather \
        --outdir gsd_results_nohidden_denovo_elbo_T12_split${SPLIT}" \
        slurm/run_nohidden_denovo_elbo.sh 12
    done

#### ELBO loss

The loss combines a binary cross-entropy on the amortisation network (tanh outputs → Bernoulli posterior) with a Besag pseudo-likelihood on the Ising prior (J, h):

    L = CE(x_denoised, x_clean) + α · PL(J, h; x_clean)

where PL is `−log p(x_i | x_{−i}) = softplus(−2 x_i (h_i + Σ_j J_ij x_j))`, controlled by `--pl-alpha` (default 0.3).

#### Noise tempering (easy → hard)

| Distribution | Beta params | Mean FN | Purpose |
|-------------|-------------|---------|---------|
| beta_low | Beta(1,5) | ~0.17 × fn_max | Gentle; builds J/h |
| beta_mid | Beta(2,3) | ~0.40 × fn_max | Moderate bridge |
| beta_high | Beta(2,1) | ~0.60 × fn_max | Biased toward LBCA/LUCA |
| beta_hard | Beta(5,1) | ~0.83 × fn_max | Extreme FN stress |
| uniform | Uniform(0, fn_max) | Full spectrum | Final fine-tuning |

#### Training stages

| Stage | Type | Params trained | FN dist | Epochs | LR | Description |
|-------|------|---------------|---------|--------|----|-------------|
| S1 | MLM | h, J | — | 100 | 1e-3 | Pseudolikelihood pre-training of core Ising params |
| S2a | MLM | A, U, W + cond + aux + τ | — | 100 | 5e-4 | Hidden-state + conditioning + module auxiliary loss |
| S2b | MLM | all (J at differential LR) | — | 200 | 3e-4 | Joint refinement of all params |
| S3a | Denoiser | ising + hidden + scalar skip + cond + τ | beta_low | 200 | 1e-4 | Core denoiser at gentle noise |
| S3b | Denoiser | same as S3a | beta_mid | 150 | 8e-5 | Moderate noise bridge |
| S3c | Denoiser | + diagonal gates | beta_high | 150 | 5e-5 | Per-gene diagonal gating |
| S3d | Denoiser | + lowrank gates (all params) | beta_hard | 150 | 3e-5 | Full model at extreme noise |
| S3e | Denoiser | gates + cond only (J frozen) | uniform | 100 | 1e-5 | Fine-tune gates across full FN spectrum |

Total: ~1150 epochs. Checkpoints and progress saved per stage; training resumes automatically from progress.json.

### Higher-order, no-hidden chain (train_higher_order_nohidden.py + launch_ho_chain.sh)

`NoHiddenHigherOrderDenoiser` is the visible-only counterpart to `HigherOrderDenoiser` — adds attention Δ(x, t) on top of `NoHiddenDenoiser` (no hidden state, no z input to attention). Same HO1/HO2/HO3 curriculum and zero-init head. The script enforces that the `--init-from` checkpoint is a `NoHiddenDenoiser` (or already a `NoHiddenHigherOrderDenoiser`) and rejects hidden sources.

Output dir: `gsd_results_higher_order_nohidden_T${T}_split${S}/`.

#### Sequential chain across (split, T)

`launch_ho_chain.sh` walks (split, T) ∈ {1..10} × {8, 12, 16, 20} = 40 jobs total. Each job picks the right `--init-from` for the target T:

| T  | `--init-from` checkpoint                                                           |
|----|------------------------------------------------------------------------------------|
| 8  | `gsd_results_nohidden_denovo_elbo_T8_split{S}/model_s3e.pth` (denovo S3e)          |
| 12 | `gsd_results_nohidden_finetune_chain_T8to20_split{S}/model_T8to12_f3.pth`          |
| 16 | `…finetune_chain_T8to20…/model_T12to16_f3.pth`                                      |
| 20 | `…finetune_chain_T8to20…/model_T16to20_f3.pth`                                      |

Note: T={12,16,20} all init from the **single chained** `T8to20` finetune log — i.e. sequential 4-step legs. The per-T `finetune_chain_T8to{12,16}` directories (if present) are not used as HO source.

The SLURM wrapper for each job (`slurm/run_higher_order_nohidden.sh`) waits for `done.flag`, then re-`sbatch`es itself with `HO_QUEUE` carrying the rest of the queue. Combined with mid-stage `save_ckpt_mid` resumes, the whole 40-job chain runs unattended.

    # Submit the chain (writes one queue, advances on done.flag):
    ./scripts/launch_ho_chain.sh

### Onsager mode persistence (checkpoint sidecar)

Each saved checkpoint writes a `model_<name>.cfg.json` sidecar alongside
`model_<name>.pth` containing `{onsager, no_hidden, T}`. The trainers default to
`--onsager auto`, which reads the value from the sidecar; an explicit `--onsager`
is validated against it (aborting on mismatch). Reused by
`analyze_ancestral_node.py::build_model_for_ckpt` to auto-detect the class.

## Calibration analysis

`calibration_analysis.py` measures posterior probability calibration at multiple biological aggregation levels (global, per-COG-category, per-KEGG-module, per-COG, and module-completeness). Uses DDP for multi-GPU prediction generation and vectorised numpy for metrics.

    # On SLURM (8 GPUs (80 GB each), ~2h):
    sbatch --export=ALL,EXTRA="--ckpt-hidden gsd_results_denovo_elbo_onsager_T8_split1/model_s3e.pth \
        --ckpt-nohidden gsd_results_nohidden_denovo_elbo_T8_split1/model_s3e.pth \
        --outdir calibration_T8_split1" \
        slurm/run_calibration.sh

Produces:
- 7 PDF figures (Cal1–Cal7): reliability diagrams, ECE heatmaps, module-tier comparisons, per-COG scatter, KEGG metabolic map overlay, calibration funnel
- 3 LaTeX tables: global metrics with bootstrap CIs, COG category summary, three-tier module summary
- 5 parquet files: global_metrics, cog_metrics, cat_metrics, module_metrics, comp_metrics

## Building the module matrix

`data/module_matrix_kegg.pt` is the conditioning matrix: 26 COG functional
categories + 66 COG pathway groupings + 327 KEGG metabolic modules = 419 columns
(COG -> KO -> module via the KEGG REST API, ~20 min).

    python scripts/build_module_matrix_kegg.py \
        --feather data/COG_train1_phylum.feather --output data/module_matrix_kegg.pt --resume

## Publication report

Upon completion, `train_denovo.py` generates a publication-quality report in `{outdir}/report/`:

    report/
    ├── precision_recall_vs_fn.pdf   — two-panel Precision + Recall vs FN
    ├── f1_vs_fn.pdf                 — F1 score vs FN
    ├── mcc_vs_fn.pdf                — MCC vs FN (primary metric)
    ├── results_report.tex           — LaTeX with progressive tables and figures
    └── spectra.json                 — raw metric data (JSON)

All figures include a null-expectation baseline (returning the noisy input unchanged) and shade the LBCA/LUCA operating region (FN = 0.75–0.90).

## Evaluation

Primary metric: **MCC at FN = 0.75–0.90** (LBCA/LUCA operating point). MCC accounts for the heavy class imbalance (most genes absent) by incorporating all four confusion matrix entries.

Multi-step inference (conditioned model only):

    from ising_denoiser.metrics import eval_spectra_multistep
    spectra = eval_spectra_multistep(model, val_t, dev, amp, M_mod, M_sizes, n_steps=3)

## Key design decisions

**Onsager/TAP correction:** Naive mean-field ignores the self-reinforcement of each spin on its own local field. The TAP correction `−x(1−x²)@J²` removes this bias, improving calibration at high FN where many genes are missing and the mean-field approximation is poorest.

**Unbounded skip gates (not sigmoid):** Sigmoid gates initialised at σ(−5) ≈ 0.007 had gradient ~0.007, making learning ~1000× too slow. Unbounded gates start at α=1 and move freely.

**V, P initialised tiny random (not zero):** At V=0, ∂(VVᵀ)/∂V = 0 — inescapable saddle point.

**Differential LR for J:** Gate gradients are much steeper than J. Without rate limiting, J inflates from ~37 to ~50+.

**Standalone aux head (not DDP submodule):** Called outside forward() — DDP would mark params "unused" then "ready twice". Standalone with add_param_group() avoids this.

**Module conditioning over τ conditioning:** τ unknown at inference. f_obs computable from any input and 419-dimensional.

**ELBO loss over wMSE:** The ELBO-derived loss combines a cross-entropy on the amortisation network with a Besag pseudo-likelihood that directly regularises J and h, providing a principled probabilistic objective.

## Coupling matrix validation against known PPIs

The coupling matrix $J\in\mathbb{R}^{N\times N}$ encodes pairwise gene
co-occurrence preferences across ~90K extant prokaryotic genomes. Whether $J$
recovers known protein-protein interactions is benchmarked against STRING v12
across the 19 ProteomeLM pathogens -- see the report's interactome section and
`scripts/interactome/` (`extract_J.py`, `map_proteome_to_cog.py`,
`benchmark_string_ppi.py`, `aggregate_interactome.py`).



## How this code was written

Much of the implementation and most of the documentation in this repository were
written with AI coding assistants -- Claude Code (Anthropic) and ChatGPT (OpenAI) --
working under human direction.

The division of labour was roughly this. The scientific design is the authors': the
formulation of ancestral gene-content reconstruction as denoising under a learned
Ising model, the architecture of the denoiser, the choice of whole-phylum hold-out as
the validation regime, the decision to report ensemble means with cross-split spread,
the analyses and the interpretation of their results. The assistants wrote a large
share of the code that implements those decisions, and drafted much of the
documentation, docstrings and comments. Debugging, experiment design, the successive
rounds of revision, and every judgement about what the results mean were human.

We note this because readers should know how the artefact in front of them was made,
and because it bears on how to read it. Code written this way can be fluent and still
be wrong in ways that fluency hides; it should be checked, not trusted. Where this
repository records a problem with its own results -- the hold-out leakage in the
fine-tuned specialists, the retracted headline recovery figure, the numbers that come
from a superseded corruption draw -- those notes are there because the authors put
them there, and they are the parts most worth reading.

## References

- Davín AA, et al. (2025). A geological timescale for bacterial evolution and oxygen adaptation. Science 388(6742), eadp1853.
- Galperin MY, et al. (2025). COG database update 2024. Nucleic Acids Research 53(D1).
- Kanehisa M, et al. KEGG for taxonomy-based analysis of gene sets.
- Malbranke C, Zalaffi GP, Bitbol A-F (2025). ProteomeLM: A proteome-scale language model. bioRxiv 2025.08.01.668221.
- Szklarczyk D, et al. (2023). The STRING database in 2023. Nucleic Acids Research 51(D1).
