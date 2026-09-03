# Training the Ising gene-content denoiser

How the models in `MODELS.md` were trained, end to end. The pipeline is a single
warm-started chain: a **de-novo plain Ising** base, a **higher-order (HO)
fine-tune** that adds attention, **domain / FP-tolerant specialists**, and then
the **focal reconstruction models** (the high-completeness + marginal-FP +
consistency retrains that the manuscript uses for every LBCA/LACA result).
Inference (reconstruction) is driven by `scripts/reconstruct.py`. NOTE: the
specialists do NOT share the generalist's whole-phylum hold-out -- each
specialist is built by an INDEPENDENT re-shuffle on a domain-filtered pool, so
its val set is decoupled from the root's. See "Hold-out integrity" below.

```
de-novo plain Ising  ->  HO fine-tune        ->  domain / FP specialists  ->  FOCAL reconstruction models
(NoHidden, T8->T20)      (add attention,          (mix-FT, bac-FT, arc-FT,     HQ retrain (_real_hq, Sec 7.1)
 Sec 1)                   HO T8..T20, Sec 2)       + mix-FT-fp, bac-FT-fp;        -> marginal-FP (_marginal_hq, Sec 7.2)
                                                   Sec 3)                          -> consistency (_cons, Sec 7.3)
                                                                                 + E. coli zoom-in ladder (Sec 8, Fig 3)
```

Each arrow is a warm-started fine-tune (`--init-from <prev>/model_ho3.pth`); every
family is x10 CV splits and resume-safe. Stages 1-3 + the feather build (Sec 6)
were already documented; Sec 7-8 below add the focal-model and zoom-in recipes.

---

## 0. Data and hold-out

- **Features.** Each genome is a length-`N = 4789` binary present/absent vector
  over orthologous gene families (COGs, from eggNOG annotations), `x_i in {-1,+1}`.
- **Pool.** ~113,000 modern genomes (~107k Bacteria across 100+ GTDB r220 phyla,
  ~6k Archaea) [Davin et al. 2025], with GTDB taxonomy + CheckM QC columns
  alongside the COG columns.
- **Hold-out: whole-phylum, 10-way.** `scripts/build_feathers.py` assigns
  *entire GTDB phyla* to train or val (never split a phylum), greedily packing
  ~20% of genomes into val, with 10 seed shuffles -> 10 replicate splits
  `data/COG_{train,val}{1..10}_phylum.feather`. This tests generalisation to
  **unseen lineages**, not interpolation within known ones -- the deep-ancestor
  setting. Domain specialists use `build_bacterial_feathers.py` /
  `build_archaeal_feathers.py`, which filter the pool to one domain and re-run
  whole-phylum hold-out (independent shuffles), with optional subsampling
  (~15k train / 3k val) to keep fine-tunes at small-drift scale. CAVEAT: that
  independent re-shuffle DECOUPLES the specialist hold-out from the root. A
  genome held out of the specialist val set can still have its phylum in the
  root `COG_train{N}` -- ~27% of bac val genomes do. The whole-phylum guarantee
  is preserved only WITHIN each feather, not jointly with the root backbone the
  specialist fine-tunes (see "Hold-out integrity" below).

**Hold-out integrity: leak in the fine-tuned specialists.** The whole-phylum
property holds only WITHIN each feather: inside a single split, a genome's
entire phylum sits in either train or val. It is NOT preserved jointly between
a specialist and the root backbone it fine-tunes. The bac-FT / mix-FT
specialists fine-tune a root (HO-T20) trained on `COG_train{N}`, an
INDEPENDENTLY shuffled pool, so a genome held out of the specialist feather can
still have its phylum in the root training set (frozen `J`, a few epochs at
1e-6 -- the fine-tune cannot un-memorise what the root saw). A whole-phylum
hold-out for a fine-tuned specialist must therefore be verified against BOTH
the specialist feather AND the root feather. The earlier *E. coli*
"verification" checked only `COG_bac_train1_phylum.feather` (where
`p__Pseudomonadota` is absent) and never checked `COG_train1_phylum.feather`,
where `p__Pseudomonadota` IS present on split 1. *E. coli*
(`RS_GCF_003697165.2`) is root-clean only on splits 5/9; `reconstruct_extant.py`
picks the split by `COG_bac_val{N}` membership alone and auto-selected split 1 =
ROOT-LEAKED. So 26.7% of the bac-FT val set is root-leaked overall (split1 93%,
split9 75%, split5 0.5%).

The headline "93% *E. coli* recovery by bac-FT" is therefore the root-LEAKED
split-1 number. Leak-free (root-clean split 5) *E. coli* recall, as the TYPICAL
noise instance (median over 500 MARGINAL-FP corruption draws; see
scripts/score_ecoli_replicates.py and Fig 2/3), is 77/72/59% at fN 0.4/0.6/0.8 (an
earlier single permissive draw read 79/71/66%; figures quoting 74/67/57% at
fN 0.5/0.75/0.9 come from that permissive draw rather than the typical run). That is PARTIAL generalisation,
not the ">90%, genuine generalisation not memorisation" claim; the specialist's
defensible, leak-free advantage over the generalist is FP-rejection ROBUSTNESS,
not recall.

---

## 1. De-novo plain Ising (`train_denovo.py`)

Trains the **NoHidden** pairwise denoiser from scratch (no pretrained
checkpoint): learned fields `h`, symmetric hollow couplings `J`, the gated
step-dependent mean-field/TAP update, adaptive temperature, and a
module-completeness conditioner -- **no attention**. This is the "Ising-only J".

- **Loss (`--loss`).** `elbo` (production): binary cross-entropy on the
  amortisation network + a Besag **pseudo-likelihood** on the Ising prior
  `(J, h)` (`--pl-alpha`), which removes the ad-hoc `pos_weight` and yields
  calibrated posterior means in `[-1,1]`. `wmse` (legacy): presence-weighted
  MSE with `w_+ = 3` to counter the ~75% absence base rate.
- **Curriculum.** An 8-stage schedule that jointly tempers two axes -- the
  false-negative corruption level and the relaxation sharpness -- so the model
  first learns easy denoising, then the deep-deletion regime.
- **Relaxation-depth chain.** The de-novo model is then deepened by warm-started
  fine-tunes that extend the unrolled depth and copy/extend the per-step gates:
  **T8 -> T12 -> T16 -> T20**
  (`gsd_results_nohidden_finetune_chain_T8to20_split*/model_T{8to12,12to16,16to20}_f*.pth`).
  More sweeps let the learned couplings propagate evidence further before the
  prediction is read out; depth helps exactly in the deep-FN tail.

**Output:** plain **Pairwise T{8,12,16,20}** (`MODELS.md`).

---

## 2. Higher-order fine-tune (`train_higher_order_nohidden.py`)

Adds a learned **third-order attention head** (1 layer, 4 heads, `d=128`,
+818,753 params) on top of the frozen pairwise base, giving the
`NoHiddenHigherOrderDenoiser`. Marginalising the head induces effective
couplings of arbitrary order among genes, capturing "a module is present only
when all its members are" -- which pairwise `J` cannot express.

Initialised from the plain chain (`--init-from .../model_T16to20_f*.pth`) and
trained in three sub-stages (loss = `elbo`):

| stage | trainable | epochs / lr |
|---|---|---|
| **HO1** | attention head only (`J`, gates, cond frozen) | 20 / 1e-4 |
| **HO2** | attention + gates + module-cond (`J` frozen) | 40 / 5e-5 |
| **HO3** | full joint (`J` at `j_lr_frac=0.1` x lr -- gentle co-adaptation) | 20 / 2e-5 |

**Output:** **HO T{8,12,16,20}** (`gsd_results_higher_order_nohidden_T{...}_split*/model_ho3.pth`).
The **HO T20** model is the **production generalist** and the source of every
fine-tune below. (A `train_add_hidden.py` variant adds explicit `H=1000` hidden
units on top -- the HO+hidden T8 control -- which adds essentially nothing over
the no-hidden head at matched depth, so it is not used downstream.)

The pairwise `J` is barely perturbed by the head: the plain-T20 and HO-T20
coupling matrices correlate at ~1.000, so "the Ising J" is well-defined across
the family.

---

## 3. Specialists (fine-tunes of HO T20)

All specialists fine-tune the HO-T20 generalist (`--init-from .../model_ho3.pth`)
on the same noise model, differing only in the training subset / curriculum:

- **mix-FT** (`..._T20_mix`): mixed-domain **rebalance** that up-weights the
  minority archaeal genomes. Used for the **archaeal LACA**. NOTE: mix-FT
  validation was AUDITED for the root leak and is CLEAN -- the spectra are scored on
  `COG_val` (phylum-disjoint from the backbone `COG_train`, 0/244,643); the
  mixed-domain fine-tune trains on `COG_mix_train` (a different split holding ~92% of
  `COG_val` phyla), but this does NOT inflate recovery: stratified clean-vs-leaked MCC
  has leaked-minus-clean NEGATIVE (-0.02), the opposite of bac-FT's +0.09..0.12 (see
  `analysis/leakage_audit/scratch_ecoli_split5/FIG1_mixFT_leak_test.md`). Launcher:
  `scripts/launch_ho_nohidden_T20_mix_finetune.sh`.
- **bac-FT** (`..._T20_bac_hard`): **bacteria-only**, with the **hard
  false-negative curriculum** (mass at fN in [0.75,0.90]). Used for the
  **bacterial LBCA**. `build_bacterial_feathers.py` + the bac FT wrapper.
  NOTE: bac-FT's reported whole-phylum MCC (0.802/0.735/0.674) and 93% *E. coli*
  recovery are ROOT-LEAK-inflated (specialist val not disjoint from root
  `COG_train{N}`; ~27% leaked). Honest clean-subset MCC is ~0.780/0.705/0.643;
  the specialist's edge over the generalist is marginal (+0.006 @fN0.9, not
  +0.037), and its real advantage is FP-rejection robustness, not recall.
- **arc-FT** (`..._T20_arc`): **archaea-only** control. Worst model on both
  the both-domain and archaea-only held-out sets -- archaea are the minority
  domain, so archaea-only fine-tuning overfits/forgets rather than specialises.
  Not used for any reconstruction.
- **FP-tolerant variants** (`-fp` suffix): **mix-FT-fp**, **bac-FT-fp**, the same
  fine-tunes plus a **false-positive curriculum** rising to `fP_max = 0.1`
  (Beta(1,5)-weighted toward low rates; `--fp-max`/`--fp-dist`). For the high-FP
  regime; see the report's "Robustness to false positives".

---

## 4. Noise model and objective (shared)

From each clean genome `x*`, a noisy input `x0` is synthesised by two channels,
matching how reconciliation corrupts ancestral input:

- **False-negative channel** (present -> absent, rate `fN`), drawn per genome
  from a difficulty curriculum
  `fN ~ Uniform[0,0.9]`, then `0.9*Beta(2,1)`, then `0.9*Beta(5,1)`
  (the last concentrating mass in the deep-ancestral tail `fN in [0.75,0.90]`).
- **False-positive channel** (absent -> present, rate `fP = 0.01`; for the
  `-fp` models a curriculum up to `0.1`).
- **Objective**: the ELBO loss above (BCE + Besag pseudo-likelihood) or the
  legacy presence-weighted MSE (`w_+ = 3`), plus a module-completeness auxiliary
  (`lambda = 0.3` in the MSE recipe).

`T` is the **relaxation depth** (number of unrolled mean-field sweeps), not a
softmax temperature -- the temperature is a separate learned, density-dependent
quantity.

---

## 5. Reproducibility

- **Train:** `scripts/build_feathers.py` (+ `build_{bacterial,archaeal}_feathers.py`,
  `build_hq_feathers.py` Sec 6.4) -> `train_denovo.py` ->
  `train_higher_order_nohidden.py` -> the specialist launchers
  (`scripts/launch_ho_nohidden_T20_*_finetune.sh`) -> the focal-model launchers
  (`slurm/run_fp_realistic_hq.sh` -> `run_fp_marginal.sh` -> `run_consistency_test.sh`,
  Sec 7) and the Fig 3 ladder (`run_ecoli_zoom_holdout.sh`, Sec 8). SLURM wrappers
  in `slurm/`; all resume-safe; 10 splits per family.
- **Checkpoints:** `gsd_results_<family>_split{1..10}/model_ho3.pth` (HO + FTs),
  `.../model_T16to20_f*.pth` (plain chain).
- **Catalog:** `MODELS.md` (params, scope, ancestral MCC/ECE per model).
- **Validation + provenance:** the MBE paper -- `analysis/mbe_manuscript.tex` +
  `analysis/SM.tex`; reproduce every figure/table via `analysis/REPRODUCE.md`
  (`scripts/build_mbe_figures.py` for figures, `scripts/build_paper_artifacts.py`
  for tables). (`analysis/report.tex` is the deprecated older report.)
- **Inference:** `python scripts/reconstruct.py --list`.

---

## 6. Data preparation: building the feathers

The training feathers are produced by three scripts, run in order. Step 1
builds the mixed-domain (generalist) pool from the raw annotations; steps 2-3
derive the domain-specialist splits from that pool (fast -- no re-pivot of the
4 GB annotation table). All three do **whole-phylum hold-out** (no phylum in
both train and val; greedy ~20% val packing; one independent shuffle per
replicate) and are deterministic given `--seed`.

### 6.1 `build_feathers.py` -- generalist pool (raw -> COG_{train,val})

Pivots the eggNOG COG annotations to a genome x COG `{-1,+1}` matrix, parses
GTDB r220 taxonomy + CheckM QC, and writes the 10 whole-phylum splits.

```
python scripts/build_feathers.py \
    --eggnog   filtered_all_eggnog.csv \      # per-genome COG annotations (~4 GB source)
    --ar-meta  ar53_metadata_r220.tsv \       # GTDB archaeal metadata (taxonomy + CheckM)
    --bac-meta bac120_metadata_r220.tsv \     # GTDB bacterial metadata
    --split-rank phylum --replicates 10 --val-frac 0.2 --seed 42 \
    --outdir data
# -> data/COG_{train,val}{1..10}_phylum.feather   (~113k genomes x 4789 COGs)
```
(The production run used `--replicates 10`; the script default is 3.)

### 6.2 `build_bacterial_feathers.py` -- bacterial specialist splits

Takes the generalist split-1 feathers (their union is the full ~113k pool),
filters to `d__Bacteria` (~107k genomes, 100+ phyla), re-runs whole-phylum
hold-out with 10 **independent** seed shuffles, and subsamples to keep the
fine-tunes small. Produces the splits for **bac-FT** (LBCA) and **bac-FT-fp**.

WARNING: these 10 bac shuffles are INDEPENDENT of the root split, so the bac val
sets are NOT root-disjoint -- ~27% of bac val genomes have their phylum in the
root `COG_train{N}` (split1 93%, split5 0.5%). For a leak-free bac-FT eval, pin
a split where the genome is held out of BOTH `COG_bac_val{N}` AND
`COG_train{N}`.

```
python scripts/build_bacterial_feathers.py \
    --source-train data/COG_train1_phylum.feather \
    --source-val   data/COG_val1_phylum.feather \
    --replicates 10 --val-frac 0.2 --seed 42 \
    --max-train 15000 --max-val 3000 --outdir data
# -> data/COG_bac_{train,val}{1..10}_phylum.feather
```

### 6.3 `build_archaeal_feathers.py` -- archaeal / mixed specialist splits

Same recipe filtered to `d__Archaea` (5,869 genomes, 21 phyla). With
`--mix-bacteria-ratio 0` it writes pure-archaea `COG_arc_*` (for **arc-FT**);
with a positive ratio it mixes a balanced fraction of bacteria back in to write
`COG_mix_*` -- the rebalanced pool for **mix-FT** / **mix-FT-fp** (the LACA
model).

```
# pure archaea (arc-FT)
python scripts/build_archaeal_feathers.py \
    --source-train data/COG_train1_phylum.feather \
    --source-val   data/COG_val1_phylum.feather \
    --replicates 10 --mix-bacteria-ratio 0.0 --outdir data
# -> data/COG_arc_{train,val}{1..10}_phylum.feather

# archaea + rebalanced bacteria (mix-FT / mix-FT-fp)
python scripts/build_archaeal_feathers.py \
    --source-train data/COG_train1_phylum.feather \
    --source-val   data/COG_val1_phylum.feather \
    --replicates 10 --mix-bacteria-ratio <r> --outdir data
# -> data/COG_mix_{train,val}{1..10}_phylum.feather
```

The raw inputs (`filtered_all_eggnog.csv` and the two GTDB metadata TSVs) are
the large gitignored source data; they are archived alongside the feathers so
step 6.1 can be reproduced from scratch (see the open-access archive record).
The feathers themselves and the final production checkpoints are hosted in the
open-access archive rather than in git (they are too large); see `ARCHIVE.md`.

### 6.4 `build_hq_feathers.py` -- high-completeness (HQ) splits

The focal reconstruction models (Sec 7) train on **near-complete** genomes, so
the clean target is a genuinely complete genome (incomplete genomes inject
false-negative label noise -- unassembled-but-present genes scored "absent" --
biasing the model to under-call). `build_hq_feathers.py` re-filters the existing
`COG_{bac,mix}_{train,val}{N}` splits by a **per-domain CheckM cut** (bacteria
completeness >= 90, archaea >= 80 to preserve deep diversity; contamination <= 5
both), keeping the whole-phylum hold-out intact.

```
python scripts/build_hq_feathers.py --fams bac mix --splits 10 --outdir data
# reads  data/COG_{bac,mix}_{train,val}{1..10}_phylum.feather  (+ CheckM columns)
# writes data/COG_{bac,mix}_hq_{train,val}{1..10}_phylum.feather
```

### 6.5 `build_ecoli_zoom_feathers.py` -- the Fig 3 leave-clade-out ladder

Re-partitions the split-5 HQ bacterial corpus by nested clades around the Fig 2
*E. coli* genome, so each rung's nearest training relative sits at a known
divergence time (phylum > class > order > intermediate > family > species).
*E. coli* itself is always held out and scored separately. Needs the per-genome
GTDB metadata table `data/genome_metadata.tsv` (shipped in the main bundle +
`extra-training-data.tar`; `--dry-run` prints the partition plan first).

```
python scripts/build_ecoli_zoom_feathers.py --ranks phylum,class,order,intermediate,family,species --outdir data
# reads  data/COG_bac_hq_{train,val}5_phylum.feather, data/COG_val5_phylum.feather, data/genome_metadata.tsv
# writes data/COG_bac_hq_ecolizoom_{rank}_{train,val}.feather  (+ data/ecolizoom_provenance.json)
```

---

## 7. Focal reconstruction models (HQ -> marginal-FP -> consistency)

These are the models the manuscript uses for **every** LBCA/LACA result. Each is
a warm-started HO fine-tune of the previous rung (same NoHidden HO-T20
architecture, `--init-from <prev>/model_ho3.pth`), x10 splits, resume-safe.

### 7.1 HQ realistic-FP retrain (`_real_hq`) -- `slurm/run_fp_realistic_hq.sh`

Re-runs the realistic coherent-FP fine-tune (Sec 3) on the **HQ feathers** (6.4),
so the clean target is a complete genome. Same schedule/init as the non-HQ FP
fine-tune; the only deltas are {HQ targets, coherent FP}.

```
for A in mix bac; do
  sbatch --export=ALL,ARM=$A,SPLIT=1,SPLIT_QUEUE="2 3 4 5 6 7 8 9 10" slurm/run_fp_realistic_hq.sh
done
#  mix: init mix-FT      -> gsd_results_higher_order_nohidden_T20_mix_fp_real_hq_split{N}
#  bac: init bac-hard-FT -> gsd_results_higher_order_nohidden_T20_bac_fp_real_hq_split{N}
```
(The `_real_hq` checkpoints are the marginal init; they are an INTERMEDIATE and are
not shipped on Zenodo -- regenerate them here from the staged `*_fp_real` models.)

### 7.2 Marginal-FP curriculum (`_marginal_hq`) -- THE FOCAL MODEL -- `slurm/run_fp_marginal.sh`

Fine-tunes `_real_hq` with a balanced K-copy curriculum per genome (same clean
target throughout): **clean** copies preserve real genomes; **rescue** copies use
high FN + low FP so the model keeps its sparse-input fill-in ability;
**marginal-prune** copies inject a dense FP drawn by per-COG cross-genome marginal
frequency (`--marginal-fp-rate/-max/-fn`), a broad low-coherence over-reconstruction
the model must TRIM back to the clean genome. The model reads the input's coupling
coherence to decide fill-in vs trim.

```
for A in bac mix; do sbatch --export=ALL,ARM=$A,SPLIT=1,SPLIT_QUEUE="2 3 4 5 6 7 8 9 10" slurm/run_fp_marginal.sh; done
#  bac: init bac_fp_real_hq -> gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split{N}   (LBCA focal)
#  mix: init mix_fp_real_hq -> gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split{N}   (LACA base)
#  reconstruct.py names these  bac-FT-fp-marginal-hq / mix-FT-fp-marginal-hq
```

### 7.3 Cross-input consistency fine-tune (`_cons`) -- `slurm/run_consistency_test.sh`

Fine-tunes the marginal model with an added cross-input **consistency loss**:
penalise the variance across the K corrupted copies so the sparse/under- and
dense/over-corrupted copies map to the SAME denoised output (input-invariance),
collapsing the sparse-build vs dense-trim fixed-point gap. Faithful default
(identical schedule + gentle LRs, `J` frozen, lambda=1); the manuscript's LACA
uses the **mix** cons model.

```
for A in mix bac; do sbatch --export=ALL,ARM=$A,LAMBDA=1.0 --array=1-10 slurm/run_consistency_test.sh; done
#  init {bac,mix}_fp_marginal_hq -> gsd_results_consistency_T20_{arm}_fp_marginal_cons_l1.0_j1.0_hq_split{N}
#  reconstruct.py names these  bac-FT-fp-cons-hq / mix-FT-fp-cons-hq
```

---

## 8. E. coli zoom-in ladder (Fig 3 -- recovery vs divergence time)

A full hard retune of the marginal-HQ regime with `J` UNFROZEN
(`--j-lr-frac 0.1`, HO2 = 20 @ 2e-5, HO3 = 12 @ 1e-5) so the couplings absorb the
added close relatives, run once per rung on the zoom feathers (6.5). Init = the
same source the published marginal-HQ split-5 used
(`bac_fp_real_hq_split5/model_ho3.pth`). *E. coli* is never trained on and is
scored by `reconstruct_extant.py`; `scripts/score_ecoli_recovery.py` +
`scripts/plot_ecoli_timeladder.py` turn the held-out recoveries into Fig 3.

```
for R in phylum class order intermediate family species; do
  sbatch --gres=gpu:2 --export=ALL,RANK=$R slurm/run_ecoli_zoom_holdout.sh
done
#  -> gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_{rank}   (shipped: ecolizoom-models.tar)
```
