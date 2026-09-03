#!/bin/bash
#SBATCH --job-name=gsd-ah-T20
#SBATCH --output=gsd-ah-T20-%j.out
#SBATCH --error=gsd-ah-T20-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#
# 4 GPUs per split (launcher runs TWO splits concurrently under the # 8-GPU assoc cap, like launch_ho_chain.sh / launch_ho_fp_finetune.sh).
#
# SLURM script for the cluster (GPU nodes, 80 GB).
# Add hidden state on top of a NoHidden HO3 T=20 checkpoint with a
# TRANSPARENT START (A=0 -> model identical to source at step 0) and a
# MODULE-COHERENT CORRUPTION CURRICULUM (inject/delete/swap whole KEGG
# modules) so the hidden state learns module co-occurrence rather than
# just absorbing per-COG residual noise.
#
# Source ensemble: gsd_results_higher_order_nohidden_T20_split{1..10}/model_ho3.pth
# Target: gsd_results_addhidden_T20_split{1..10}/model_ho3.pth
#
# Full TAP onsager, ELBO loss (ce + 0.3*pl), tied attention (HO_tied default).
# Stages:
# AH1 hidden only (A, U, W); J + gates + attn + cond FROZEN; heavy
# module corruption; aux_lambda x 3 to force z to encode module
# signatures. ~12 ep @ 1e-4.
# AH2 hidden + gates + cond + attn; J FROZEN; mixed noise. ~15 ep @ 5e-5.
# AH3 full joint; j-lr-frac 0.1 protection on J. ~8 ep @ 2e-5.
#
# Wall-time note: full bacterial feathers (~90k genomes), 35 epochs at
# T=20, 4 GPUs. Comparable to a single nohidden HO T=20 leg in
# launch_ho_chain (~28 ep -> ~10-12 h). Expect ~10-14 h per split with
# at most one auto-resubmit.
#
# Usage (single split):
# sbatch --export=ALL,EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth \
# --train-feather data/COG_train1_phylum.feather \
# --val-feather data/COG_val1_phylum.feather \
# --outdir gsd_results_addhidden_T20_split1" \
# slurm/run_ho_addhidden_T20.sh
#
# Normally launched via scripts/launch_ho_addhidden_T20.sh (2 x 4-GPU chains).

echo "=========================================="
echo "AddHidden T=20 - Job $SLURM_JOB_ID - $(hostname)"
echo "Started: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Prefer the project venv: its python3.11 is a symlink into the cluster
# module python install on shared FS, so it works even when 'module load
# python/3.11.11' fails because of unrelated Lmod dependencies.
if [ -f .venv/bin/activate ]; then
 source .venv/bin/activate
 echo "Activated venv python: $(command -v python3.11) -> $(readlink -f "$(command -v python3.11)")"
else
 echo "WARNING: .venv missing -- falling back to module load python/3.11.11"
 [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
 module load python/3.11.11
fi

NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES} (${NGPUS} GPUs)"

OUTDIR=$(echo "${EXTRA}" | grep -oE -- '--outdir[[:space:]]+[^[:space:]]+' | awk '{print $2}')
OUTDIR=${OUTDIR:-gsd_results_addhidden_T20}
echo "Outdir: ${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"
echo "EXTRA: ${EXTRA}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

# BPG=128 for T=20 with hidden state: comparable to
# run_higher_order_nohidden.sh's T=20 BPG=128 (~33 GiB / GPU at no-hidden).
# Adding hidden bumps memory ~10-15%; 128 leaves headroom.
BPG=${BPG:-128}

AH1_EP=${AH1_EP:-12}
AH2_EP=${AH2_EP:-15}
AH3_EP=${AH3_EP:-8}
echo "Schedule: AH1=${AH1_EP} AH2=${AH2_EP} AH3=${AH3_EP} BPG=${BPG}"

# Pre-flight: the NoHidden HO3 source must exist (else exit cleanly so
# the resubmit chain retries -- the source could still be training).
INIT_FROM=$(echo "${EXTRA}" | grep -oE -- '--init-from[[:space:]]+[^[:space:]]+' | awk '{print $2}')
if [ -n "${INIT_FROM}" ] && [ ! -f "${INIT_FROM}" ]; then
 echo "[wait] --init-from ${INIT_FROM} not found yet - will resubmit and retry"
 mkdir -p ${OUTDIR}
 [ ! -f ${OUTDIR}/progress.json ] && \
 echo '{"completed":[],"waiting_for_checkpoint":true}' > ${OUTDIR}/progress.json
 PY_EXIT=1
else

# Note: --onsager is hard-coded to 'full' inside train_add_hidden.py (per
# the "full TAP" design choice). Attention args must match HO_tied:
# d_model=128, nhead=4, n_layers=1, dim_ff=512.
python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
 train_add_hidden.py \
 --new-T 20 \
 --module-matrix data/module_matrix_kegg.pt \
 --ah1-epochs ${AH1_EP} --ah1-lr 1e-4 \
 --ah2-epochs ${AH2_EP} --ah2-lr 5e-5 \
 --ah3-epochs ${AH3_EP} --ah3-lr 2e-5 \
 --j-lr-frac 0.1 \
 --attn-d-model 128 --attn-nhead 4 --attn-n-layers 1 --attn-dim-ff 512 \
 --attn-lr-frac 0.3 \
 --K 4 --batch-per-gpu ${BPG} \
 --loss elbo --pl-alpha 0.3 \
 --aux-lambda 0.3 --ah1-aux-mult 3.0 \
 --fn-max 0.9 --fp 0.01 \
 --ah1-mod-inject 0.30 --ah1-mod-delete 0.15 --ah1-mod-swap 0.55 \
 --ah23-mod-inject 0.20 --ah23-mod-delete 0.10 --ah23-mod-swap 0.30 \
 --compile \
 --ckpt-every 5 \
 ${EXTRA} 2>&1

PY_EXIT=$?
fi # end of init-from exists check

echo "Python exit: ${PY_EXIT:-0}"
echo "Finished: $(date)"

# -- Auto-resubmit until done.flag --
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-6}

if [ -f "${OUTDIR}/done.flag" ]; then
 echo "[done] done.flag present - AddHidden complete for this split"
 [ -f "${OUTDIR}/spectra_2d.tsv" ] && echo "[done] 2D spectra: ${OUTDIR}/spectra_2d.tsv"

 # Advance AH_QUEUE if there are more splits to run.
 # AH_QUEUE format: space-separated split numbers, e.g. "2 3 4 5".
 if [ -n "${AH_QUEUE}" ]; then
 NEXT=$(echo "${AH_QUEUE}" | awk '{print $1}')
 REST=$(echo "${AH_QUEUE}" | cut -d' ' -f2-)
 [ "${REST}" = "${AH_QUEUE}" ] && REST="" # only 1 item left
 NEXT_EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split${NEXT}/model_ho3.pth --train-feather data/COG_train${NEXT}_phylum.feather --val-feather data/COG_val${NEXT}_phylum.feather --outdir gsd_results_addhidden_T20_split${NEXT}"
 echo "Advancing AH queue: split=${NEXT} (${REST:-last})"
 sbatch \
 --export=ALL,EXTRA="${NEXT_EXTRA}",AH_QUEUE="${REST}",AH1_EP=${AH1_EP},AH2_EP=${AH2_EP},AH3_EP=${AH3_EP},BPG=${BPG} \
 "$0"
 fi
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
 echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up."
elif [ "${PY_EXIT}" -ne 0 ] && [ ! -f "${OUTDIR}/progress.json" ]; then
 echo "Python exited ${PY_EXIT} with no progress.json - config error, not resubmitting."
else
 NEW_COUNT=$((RESUBMIT_COUNT + 1))
 echo "done.flag missing - resubmitting (chain ${NEW_COUNT}/${MAX_RESUBMITS})"
 sbatch \
 --export=ALL,EXTRA="${EXTRA}",AH_QUEUE="${AH_QUEUE}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS},AH1_EP=${AH1_EP},AH2_EP=${AH2_EP},AH3_EP=${AH3_EP},BPG=${BPG} \
 "$0"
fi
