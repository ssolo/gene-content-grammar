#!/bin/bash
#SBATCH --job-name=spectra-sweep
#SBATCH --output=spectra-sweep-%j.out
#SBATCH --error=spectra-sweep-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#
# Broad denoising-spectra + calibration sweep -- ATOMIC 1-GPU worker.
#
# This is the per-unit worker: ONE single-GPU process scores the
# (family, split) checkpoints it is given over the 2D (FN, FP) noise grid
# (FN in 0.05 steps over [0.00, 0.90], FP in {0.01, 0.02, 0.05, 0.10}) and
# from ONE forward pass per cell writes both denoising spectra
# (MCC/F1/prec/rec/fp_removed) and calibration (ECE/MCE/Brier/NLL) into one
# parquet per checkpoint:
#
# ${OUTROOT}/${tag}_split${N}.parquet
#
# Plus a model-free noisy-input null baseline per split:
# ${OUTROOT}/null_split${N}.parquet
#
# WHY 1 GPU (no torchrun): the driver's single-GPU path runs the spectra
# noise scan as a plain gradient-free forward -- load the weights, run ONE
# forward per (FN, FP) cell on the GPU, move the (T, B, N) trajectory to the
# host and score it with the numpy spectra/calib metrics. No torch.compile,
# no Triton; no NCCL all_gathers, no per-step rank-0 numpy over 19.6M-element
# arrays, no barrier() with 7 idle GPUs. Atomic 1-GPU jobs queue fast
# (backfill into any single free GPU under the per-partition cap).
# Fan these out with scripts/launch_spectra_sweep.sh (one job per unit).
#
# Single-GPU == DDP rank-0 bit-for-bit here: noise is drawn on the full
# seeded tensor before any sharding, and the model couples genes only WITHIN
# a genome (no cross-genome batch coupling, no batchnorm), so per-genome
# trajectories do not depend on batch composition. metrics_gpu also matches
# the numpy calib_metrics bin semantics (np.digitize right=False ==
# torch.bucketize right=True), so parquets from this path are directly
# comparable to any produced by the legacy DDP path.
#
# MODEL FAMILIES (all use model_ho3.pth; T and onsager auto-detected from the
# model_*.cfg.json sidecar; tied_T8 has no sidecar and correctly defaults to
# onsager=tied for the hidden higher-order class):
# nohidden_HO_T8/T12/T16/T20 pre arch-fine-tune higher-order (NoHidden+attn)
# nohidden_HO_T20_mixFT mixed-domain (archaea+bacteria) fine-tune
# nohidden_HO_T20_arcFT archaea-only fine-tune
# hidden_HO_T8 hidden higher-order (A/W + attn), onsager tied
#
# EVAL SET: data/${VAL_PREFIX}val${N}_phylum.feather. Default VAL_PREFIX=COG_
# is the canonical split-N global held-out (both domains). For the LBCA
# (bacteria-only) eval pass use VAL_PREFIX=COG_bac_. CAVEAT: the T20
# mixFT/arcFT models were trained on a SEPARATE archaea re-partition that
# overlaps COG_val{N}, so their COG_val{N} numbers are mildly optimistic --
# the clean pre-vs-FT held-out comparison lives in the calibration launchers.
# Set VAL_DOMAIN=d__Archaea to restrict to archaea (passed to the driver).
#
# RESUME-SAFETY: a (family, split) is skipped if its parquet already exists,
# so re-runs converge. This script only READS checkpoints and WRITES
# per-checkpoint parquets; it never rm's or scancel's anything. A bounded
# auto-resubmit (MAX_RESUBMITS) covers preemption / wall-clock exhaustion,
# scoped to whatever SPLITS/FAMILIES this worker was given.
#
# USAGE -- fan out atomically (the intended path):
# scripts/launch_spectra_sweep.sh # one 1-GPU job per (family,split)
#
# USAGE -- a single unit by hand:
# sbatch --export=ALL,SPLITS=1,FAMILIES="gsd_results_higher_order_nohidden_T20:nohidden_HO_T20" slurm/run_spectra_sweep.sh
#
# USAGE -- everything on ONE GPU sequentially (debug only; slow):
# sbatch slurm/run_spectra_sweep.sh

echo "=========================================="
echo "Spectra+calibration sweep (1-GPU worker) - Job $SLURM_JOB_ID - $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Started: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Prefer the project venv (its python3.11 symlinks into the cluster module
# python on shared FS; avoids broken Lmod dependency chains on some nodes).
if [ -f .venv/bin/activate ]; then
 source .venv/bin/activate
 echo "Activated venv python: $(command -v python3.11) -> $(readlink -f "$(command -v python3.11)")"
else
 echo "WARNING: .venv missing -- falling back to module load python/3.11.11"
 [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
 module load python/3.11.11
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPLITS=${SPLITS:-"1 2 3 4 5 6 7 8 9 10"}
MODELFILE=${MODELFILE:-model_ho3.pth}
OUTROOT=${OUTROOT:-spectra_sweep}
# VAL_PREFIX selects the held-out feather family: the val path is
# data/${VAL_PREFIX}val${N}_phylum.feather
# Default "COG_" -> data/COG_val${N}_phylum.feather (mixed-domain held-out).
# LBCA bacteria-only spectra eval: VAL_PREFIX=COG_bac_ ->
# data/COG_bac_val${N}_phylum.feather (bacteria-only, held-out phyla).
VAL_PREFIX=${VAL_PREFIX:-COG_}
NGENOMES=${NGENOMES:-4096}
# Single-GPU full-batch: 4096 genomes split into EVAL_BATCH chunks for the
# forward; the (T,B,N) trajectory for the full 4096 lives on the GPU for
# metrics_gpu (~1.6 GB at T=20). 1024 is comfortable on an 80 GB GPU; if a
# node OOMs, override EVAL_BATCH=512.
EVAL_BATCH=${EVAL_BATCH:-1024}
SAMPLE_SEED=${SAMPLE_SEED:-0}
FN_STEP=${FN_STEP:-0.05}
FP_GRID=${FP_GRID:-"0.01 0.02 0.05 0.1"}
# COMPILE=0 (default) -> plain EAGER forward: no torch.compile, no Triton --
# just load the weights and run the gradient-free forward to measure spectra.
# COMPILE=1 opts back into torch.compile (Triton) if ever wanted.
COMPILE=${COMPILE:-0}
COMPILE_ARG="--compile"
[ "${COMPILE}" = "0" ] && COMPILE_ARG="--no-compile"
# PARTITION + GPU_GRES: where this worker runs, reused VERBATIM on auto-resubmit
# so a resubmitted job lands on the SAME partition/hardware the dispatcher chose
# (e.g. --partition=gpu --gres=gpu:1 to backfill the a 16 GB GPU nodes under the
# separate per-partition gpu cap) instead of reverting to the #SBATCH
# defaults baked in at the top of this file. Default = the #SBATCH defaults.
PARTITION=${PARTITION:-}
GPU_GRES=${GPU_GRES:-gpu:1}
# Dispatcher knobs (scripts/launch_spectra_sweep.sh): NULL_ONLY=1 does only the
# model-free null baseline (run as an uncapped CPU job, gres=gpu:0); SKIP_NULL=1
# does only the model families (so the 7 per-split GPU unit-jobs do not race to
# write the same null parquet). Default 0/0 -> bare run does both (debug path).
NULL_ONLY=${NULL_ONLY:-0}
SKIP_NULL=${SKIP_NULL:-0}
# VAL_DOMAIN: empty (default) = global both-domain eval; or d__Archaea.
VAL_DOMAIN=${VAL_DOMAIN:-}
DOMAIN_ARGS=""
[ -n "${VAL_DOMAIN}" ] && DOMAIN_ARGS="--val-domain ${VAL_DOMAIN}"

# family list entries are "dirprefix:tag"
FAMILIES=${FAMILIES:-"\
gsd_results_higher_order_nohidden_T8:nohidden_HO_T8 \
gsd_results_higher_order_nohidden_T12:nohidden_HO_T12 \
gsd_results_higher_order_nohidden_T16:nohidden_HO_T16 \
gsd_results_higher_order_nohidden_T20:nohidden_HO_T20 \
gsd_results_higher_order_nohidden_T20_mix:nohidden_HO_T20_mixFT \
gsd_results_higher_order_nohidden_T20_arc:nohidden_HO_T20_arcFT \
gsd_results_higher_order_tied_T8:hidden_HO_T8"}

mkdir -p "${OUTROOT}"
echo "SPLITS=${SPLITS}"
echo "OUTROOT=${OUTROOT} NGENOMES=${NGENOMES} EVAL_BATCH=${EVAL_BATCH} COMPILE=${COMPILE} FN_STEP=${FN_STEP} FP_GRID='${FP_GRID}' VAL_DOMAIN='${VAL_DOMAIN}' VAL_PREFIX='${VAL_PREFIX}'"
echo "FAMILIES=${FAMILIES}"

for N in ${SPLITS}; do
 VAL=data/${VAL_PREFIX}val${N}_phylum.feather

 echo ""
 echo "------------------------------------------------------------"
 echo "Split ${N} ($(date))"

 if [ ! -f "${VAL}" ]; then
 echo "[skip] split ${N}: missing val feather ${VAL}"
 continue
 fi

 # -- noisy-input null baseline (model-free, single process, CPU path) --
 NULL_OUT=${OUTROOT}/null_split${N}.parquet
 if [ "${SKIP_NULL}" = "1" ]; then
 :
 elif [ -f "${NULL_OUT}" ]; then
 echo "[skip] null split ${N}: ${NULL_OUT} present"
 else
 echo "=== null baseline split ${N} -> ${NULL_OUT} ==="
 python3.11 scripts/sweep_spectra_calibration.py --null-only \
 --val-feather "${VAL}" --module-matrix data/module_matrix_kegg.pt \
 --model-tag noisy --family noisy --split ${N} \
 --fn-step ${FN_STEP} --fp-grid ${FP_GRID} \
 --n-genomes ${NGENOMES} --sample-seed ${SAMPLE_SEED} \
 ${DOMAIN_ARGS} --out "${NULL_OUT}" 2>&1
 fi

 if [ "${NULL_ONLY}" = "1" ]; then
 continue
 fi

 # -- one single-GPU process per (family, split) --
 for ENTRY in ${FAMILIES}; do
 PREFIX=${ENTRY%%:*}
 TAG=${ENTRY##*:}
 CKPT=${PREFIX}_split${N}/${MODELFILE}
 OUT=${OUTROOT}/${TAG}_split${N}.parquet

 if [ -f "${OUT}" ]; then
 echo "[skip] ${TAG} split ${N}: ${OUT} present"
 continue
 fi
 if [ ! -f "${CKPT}" ]; then
 echo "[skip] ${TAG} split ${N}: missing checkpoint ${CKPT}"
 continue
 fi

 echo "=== ${TAG} split ${N} -> ${OUT} ($(date)) ==="
 # NO torchrun: plain single-process invocation -> the driver's
 # single-GPU forward-only path (host-scored numpy metrics, eager).
 python3.11 scripts/sweep_spectra_calibration.py \
 --ckpt "${CKPT}" \
 --val-feather "${VAL}" \
 --module-matrix data/module_matrix_kegg.pt \
 --model-tag "${TAG}" --family "${TAG}" --split ${N} \
 --fn-step ${FN_STEP} --fp-grid ${FP_GRID} \
 --n-genomes ${NGENOMES} --sample-seed ${SAMPLE_SEED} --eval-batch ${EVAL_BATCH} \
 ${COMPILE_ARG} ${DOMAIN_ARGS} \
 --out "${OUT}" 2>&1
 RC=$?
 if [ "${RC}" -eq 0 ] && [ -f "${OUT}" ]; then
 echo "[done] ${TAG} split ${N} ($(date))"
 else
 echo "[warn] ${TAG} split ${N}: rc=${RC}, parquet missing - will retry on resubmit"
 fi
 done
done

echo ""
echo "=========================================="
echo "Loop finished: $(date)"

# -- Bounded auto-resubmit until every (family, split) parquet exists --
# (covers preemption / wall-clock exhaustion; re-runs skip existing parquets,
# so it converges. Non-destructive: only sbatch, never rm/scancel.)
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-3}
all_done=1
for N in ${SPLITS}; do
 if [ "${NULL_ONLY}" = "1" ]; then
 # this worker only owes the null parquet(s) for its splits
 [ -f "${OUTROOT}/null_split${N}.parquet" ] || all_done=0
 continue
 fi
 for ENTRY in ${FAMILIES}; do
 TAG=${ENTRY##*:}
 PREFIX=${ENTRY%%:*}
 # only require a parquet where the checkpoint actually exists
 [ -f "${PREFIX}_split${N}/${MODELFILE}" ] || continue
 [ -f "${OUTROOT}/${TAG}_split${N}.parquet" ] || all_done=0
 done
done

if [ "${all_done}" -eq 1 ]; then
 echo "[all-done] every owed (family, split) parquet is present"
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
 echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up (some incomplete)."
else
 NEW_COUNT=$((RESUBMIT_COUNT + 1))
 echo "Not all done - resubmitting (chain ${NEW_COUNT}/${MAX_RESUBMITS}) on ${PARTITION}/${GPU_GRES}"
 # Reuse the SAME partition + GRES this worker ran on (forwarded by the
 # dispatcher) so a resubmit backfills the same hardware -- e.g. stays on
 # gpu/a 16 GB GPU instead of reverting to the #SBATCH defaults baked in
 # at the top of this file.
 sbatch \
 --partition="${PARTITION}" \
 --gres="${GPU_GRES}" \
 --export=ALL,SPLITS="${SPLITS}",FAMILIES="${FAMILIES}",MODELFILE="${MODELFILE}",OUTROOT="${OUTROOT}",NGENOMES=${NGENOMES},EVAL_BATCH=${EVAL_BATCH},COMPILE=${COMPILE},NULL_ONLY=${NULL_ONLY},SKIP_NULL=${SKIP_NULL},SAMPLE_SEED=${SAMPLE_SEED},FN_STEP=${FN_STEP},FP_GRID="${FP_GRID}",VAL_DOMAIN="${VAL_DOMAIN}",VAL_PREFIX="${VAL_PREFIX}",PARTITION="${PARTITION}",GPU_GRES="${GPU_GRES}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS} \
 "$0"
fi
