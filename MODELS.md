# Model catalog -- Ising gene-content denoiser

Registry of every trained denoiser in this project: architecture, parameter
count, intended scope, a one-line validation note, and the **ancestral-regime**
recovery/calibration (MCC and ECE at the deep-ancestral operating point
**fN = 0.90, fP = 0.01**, the point the LBCA/LACA reconstructions sit at).

Conventions
-----------
- **Vocabulary:** N = 4,789 COG gene families; spin x_i in {-1,+1}.
- **T** = relaxation depth (number of unrolled mean-field/TAP sweeps), NOT a
  temperature.
- **Hold-out:** all validation is **phylum-level** -- the 10-way CV split
  removes whole phyla, so every validation genome belongs to a phylum unseen
  in training. MCC/ECE below are 10-split means on the held-out set.
- **Ancestral MCC/ECE:** MCC (Matthews correlation; +1 perfect, 0 chance;
  higher better) and ECE (expected calibration error; 0 perfect, <0.05
  well-calibrated; lower better) at fN = 0.90, fP = 0.01, on the global
  both-domain held-out set unless noted. Source: `data/spectra_calibration_all_summary.csv`
  and report Table 1.
- **Checkpoints:** `gsd_results_<...>_split{1..10}/model_ho3.pth`
  (`model_s3e.pth` for the plain de-novo T8 base).

Architectures
-------------
- **Plain pairwise Ising** (`NoHiddenDenoiser`, no higher-order term): learned
  fields h + symmetric couplings J, gated step-dependent TAP update, adaptive
  temperature, module-completeness conditioning. No attention head.
- **Ising + higher-order** (`NoHiddenHigherOrderDenoiser`): the plain model
  plus a learned third-order attention head (d=128, 1 layer, 4 heads;
  +818,753 params). **Production architecture.**
- **Ising + hidden + higher-order** (`HigherOrderDenoiser`): adds explicit
  hidden units (H = 1000) on top of the attention head. Heavier; no recovery
  gain over the no-hidden head at matched depth (hidden units redundant).

Catalog
-------

| Model | Class | Params | T | Intended scope | Anc. MCC | Anc. ECE | Validation note |
|---|---|---:|---:|---|---:|---:|---|
| Pairwise T8  | NoHidden            | 24.97M | 8  | pairwise baseline | 0.620 | 0.048 | strong, best-calibrated at low-mid noise |
| Pairwise T12 | NoHidden            | 25.94M | 12 | pairwise baseline | 0.627 | 0.050 | depth helps the deep tail |
| Pairwise T16 | NoHidden            | 26.92M | 16 | pairwise baseline | 0.631 | 0.048 | -- |
| Pairwise T20 | NoHidden            | 27.90M | 20 | pairwise baseline | 0.632 | 0.049 | best plain depth |
| HO+hidden T8 | HigherOrder      | 31.72M | 8  | architecture test | 0.637 | 0.046 | hidden units add nothing over the no-hidden head |
| HO T8     | NoHidden+HO         | 25.79M | 8  | production family | 0.632 | 0.046 | -- |
| HO T12    | NoHidden+HO         | 26.76M | 12 | production family | 0.636 | 0.047 | -- |
| HO T16    | NoHidden+HO         | 27.74M | 16 | production family | 0.639 | 0.048 | -- |
| **HO T20 (generalist)** | NoHidden+HO | 28.72M | 20 | both-domain generalist; source of all fine-tunes | 0.640 | 0.047 | production pre-fine-tune model |
| **mix-FT T20**  | NoHidden+HO         | 28.72M | 20 | best-validated generalist; **LACA** reconstruction | **0.657** | **0.036** | best recovery + best ancestral calibration (mixed-domain rebalance) |
| arc-FT T20      | NoHidden+HO         | 28.72M | 20 | archaea-only fine-tune (control; NOT used for any reconstruction) | 0.618 | 0.054 | worst on both-domain data AND worst on archaea-only held-out (fN=0.9 MCC 0.756 vs mix-FT 0.761 vs generalist 0.772); archaea are the minority domain so arc-only overfits/forgets -- the LACA uses mix-FT, not arc-FT |
| **bac-FT T20** | NoHidden+HO     | 28.72M | 20 | bacterial specialist; **LBCA** reconstruction | ~0.66-0.67* (LEAK: regen) | ties gen.* | *scored on `COG_bac_val` ONLY, which never checks the root: ~27% of bac val genomes are root-leaked, so "unseen phyla" is NOT guaranteed (phyla may be root-seen). Leak-free (root-clean) MCC is 0.780/0.705/0.643 at fN 0.5/0.75/0.9; the 0.703@fN0.85 cell is not exactly recomputed -- honest interpolation ~0.66-0.67. The edge over the generalist nearly vanishes (~+0.006 @fN0.9); the defensible advantage is FP-rejection robustness, not recall. |
| mix-FT-fp T20      | NoHidden+HO         | 28.72M | 20 | FP-tolerant generalist (high-FP section) | 0.655 | 0.036 | REALISTIC coherent-FP curriculum (per-COG FP to fP_max=0.1 Beta(1,5), PLUS whole foreign modules grafted from other genomes); production ckpts `*_T20_mix_fp_real`. At fP=0.01 recall-neutral vs mix-FT (MCC/ECE above approximately match mix-FT); the real gain is on the coherent-contamination test, where it rejects markedly more grafted FP than the default/uniform curricula. See report "Robustness to false positives". |
| bac-FT-fp T20  | NoHidden+HO         | 28.72M | 20 | FP-tolerant bacterial specialist (high-FP LBCA) | ~bac-FT | ~bac-FT | REALISTIC coherent-FP bacterial specialist; production ckpts `*_T20_bac_fp_real`, 10/10 splits; the FP-tolerant LBCA model. Recall-neutral vs bac-FT at fP=0.01; rejects coherent contamination better than default/uniform at matched recall. |

\* bac-FT is evaluated on a bacteria-only held-out set, so its numbers are not
directly comparable to the both-domain rows. The "unseen phyla" assertion is
FALSE as stated: the eval selects splits by the bac fine-tune feather alone,
which does not check the ROOT pretrain, so a held-out phylum can still be in the
root pool (~27% of bac val genomes are root-leaked). The honest leak-free MCC is
~0.66-0.67 at fN=0.85 (LEAK: regen for the exact cell), and the ECE comparison
is no longer a clean whole-phylum result. vhard curriculum (deeper FN) was
rejected: it overshoots and regresses calibration in the deep tail.

Selection summary
-----------------
- **Production architecture:** HO (no-hidden higher-order) -- matches the
  hidden model at lower cost.
- **Calibration, not recovery, decides** the ancestral choice: plain pairwise
  wins the easy regime, mix-FT wins the deep-ancestral regime.
- **Per-node reconstruction models:** mix-FT for the LACA, bac-FT for the
  LBCA. bac-FT's "best-validated" status rests on a LEAKED validation (bac val
  set not disjoint from the root pretrain pool; ~27% root-leaked); on a clean,
  root-disjoint hold-out its edge over the generalist is marginal (+0.006
  @fN0.9), so it is chosen for FP-rejection robustness, not a clean recovery
  win. mix-FT (LACA) was AUDITED for the same root leak and is CLEAN: scored on
  `COG_val` (disjoint from the backbone `COG_train`); the fine-tune's phylum exposure
  does not inflate recovery (stratified clean-vs-leaked MCC leaked-minus-clean is
  negative, -0.02; see `analysis/leakage_audit/scratch_ecoli_split5/FIG1_mixFT_leak_test.md`).
- **High-FP (false-positive-tolerant) line:** mix-FT-fp and its bacterial
  specialist bac-FT-fp. The production checkpoints are the REALISTIC coherent-FP
  retrains (`*_fp_real`): per-COG FP plus whole foreign modules grafted from other
  genomes, modelling the coherent contamination reconciliation actually produces
  (lateral transfer / paralogue mis-assignment). The earlier per-COG-only
  uniform-FP fine-tunes (`*_T20_mix_fp` / `*_T20_mix_fp_bac`) are retained for
  comparison (reconstruct.py exposes them as `*-fp-uniform`) but superseded.
  See the report's "Robustness to false positives" section.
