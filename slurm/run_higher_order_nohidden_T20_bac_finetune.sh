#!/bin/bash
#SBATCH --job-name=gsd-ho-nh-bac-T20
#SBATCH --output=gsd-ho-nh-bac-T20-%j.out
#SBATCH --error=gsd-ho-nh-bac-T20-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#
# 4 GPUs per split (launcher runs TWO splits concurrently under the # 8-GPU assoc cap, exactly like the mix + arc + AH launchers).
#
# SLURM script for the cluster (GPU nodes, 80 GB).
# Fine-tune a NoHidden HO3 T=20 model on BACTERIA-ONLY data (the LBCA
# specialist), with an honest bacteria-only held-out val set (unseen
# bacterial phyla). Goal: better calibration (ECE) at the high-FN
# (0.75-0.90) deep-ancestor operating point.
#
# This is the bacterial mirror of run_higher_order_nohidden_T20_mix_finetune.sh.
# It reuses that script's PROVEN small-drift recipe verbatim (HO1 skipped,
# HO2=5 ep@5e-6, HO3=3 ep@1e-6, J FROZEN via --j-lr-frac 0.0) -- the recipe
# that IMPROVED high-FN ECE, as opposed to the larger arc-only fine-tune
# (HO2=10@1e-5, HO3=10@5e-6, J at 0.1) which regressed.
#
# DIFFERENCES vs run_higher_order_nohidden_T20_mix_finetune.sh:
# - data: data/COG_bac_{train,val}{N}_phylum.feather (build via
# scripts/build_bacterial_feathers.py)
# - CURRICULUM env selects the FN curriculum A/B arm:
# hard (default) -> --ho-fn-curriculum beta_hard --fn-max 0.90
# (the proven mix-FT curriculum; mean FN ~0.75)
# vhard -> --ho-fn-curriculum beta_vhard --fn-max 0.95
# (Beta(8,1); concentrates training mass at
# FN 0.85-0.95, the LBCA operating point)
# - outputs: gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split{N}/
#
# Source : gsd_results_higher_order_nohidden_T20_split{1..10}/model_ho3.pth
# (the SAME pre-FT source the mix fine-tune used)
# Target : gsd_results_higher_order_nohidden_T20_bac_{hard,vhard}_split{1..10}/model_ho3.pth
#
# Schedule (small-drift defaults; J FROZEN via --j-lr-frac 0.0):
# HO1: 0 ep (skipped; attention already warm)
# HO2: 5 ep at 5e-6 (override via HO2_EP)
# HO3: 3 ep at 1e-6 (override via HO3_EP)
# BPG: 32 (override via BPG)
#
# Wall-time estimate per split: ~25-40 min (subsampled bacteria, ~15k
# train/split via build_bacterial_feathers.py --max-train). 20 runs
# (10 splits x {hard, vhard}) across 2 concurrent chains -> a few hours
# per arm.
#
# Usage (single split, one arm):
# sbatch --export=ALL,CURRICULUM=hard,EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth \
# --train-feather data/COG_bac_train1_phylum.feather \
# --val-feather data/COG_bac_val1_phylum.feather \
# --outdir gsd_results_higher_order_nohidden_T20_bac_hard_split1" \
# slurm/run_higher_order_nohidden_T20_bac_finetune.sh
#
# Normally launched via scripts/launch_ho_nohidden_T20_bac_finetune.sh.

echo "=========================================="
echo "NoHidden HO T=20 BACTERIA-ONLY fine-tune - Job $SLURM_JOB_ID - $(hostname)"
echo "Started: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Prefer the project venv (avoids broken module-load chains on some nodes).
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

# Curriculum A/B arm: hard (proven) | vhard (deep-ancestor stress).
CURRICULUM=${CURRICULUM:-hard}
case "${CURRICULUM}" in
 hard) CURR_ARGS="--ho-fn-curriculum beta_hard --fn-max 0.90" ;;
 vhard) CURR_ARGS="--ho-fn-curriculum beta_vhard --fn-max 0.95" ;;
 *) echo "ERROR: unknown CURRICULUM='${CURRICULUM}' (want hard|vhard)"; exit 2 ;;
esac
echo "Curriculum: ${CURRICULUM} (${CURR_ARGS})"

OUTDIR=$(echo "${EXTRA}" | grep -oE -- '--outdir[[:space:]]+[^[:space:]]+' | awk '{print $2}')
OUTDIR=${OUTDIR:-gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}}
echo "Outdir: ${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"
echo "EXTRA: ${EXTRA}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

# BPG=32 (override-able): ~15k bacteria/split (build_bacterial_feathers.py
# --max-train default) on 4 GPUs gives ~100+ updates/epoch, plenty for the
# small-LR drift to be smooth over the 8-epoch total (HO2=5 + HO3=3).
BPG=${BPG:-32}

HO2_EP=${HO2_EP:-5}
HO3_EP=${HO3_EP:-3}
echo "Schedule: HO1=0 HO2=${HO2_EP} HO3=${HO3_EP} BPG=${BPG}"

# Pre-flight: the NoHidden HO3 source checkpoint must exist (else exit
# cleanly so the resubmit chain retries -- the source could still be
# training on the cluster).
INIT_FROM=$(echo "${EXTRA}" | grep -oE -- '--init-from[[:space:]]+[^[:space:]]+' | awk '{print $2}')
if [ -n "${INIT_FROM}" ] && [ ! -f "${INIT_FROM}" ]; then
 echo "[wait] --init-from ${INIT_FROM} not found yet - will resubmit and retry"
 mkdir -p ${OUTDIR}
 [ ! -f ${OUTDIR}/progress.json ] && \
 echo '{"completed":[],"waiting_for_checkpoint":true}' > ${OUTDIR}/progress.json
 PY_EXIT=1
else

# --onsager intentionally omitted: train_higher_order_nohidden.py
# auto-resolves it from the source checkpoint's .cfg.json sidecar.
# --aux-lambda 0.0 because NoHidden has no hidden state to project module
# completeness from.
python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
 train_higher_order_nohidden.py \
 --module-matrix data/module_matrix_kegg.pt \
 --new-T 20 \
 --ho1-epochs 0 \
 --ho2-epochs ${HO2_EP} --ho2-lr 5e-6 \
 --ho3-epochs ${HO3_EP} --ho3-lr 1e-6 \
 --j-lr-frac 0.0 \
 ${CURR_ARGS} \
 --attn-d-model 128 --attn-nhead 4 --attn-n-layers 1 --attn-dim-ff 512 \
 --attn-lr-frac 0.3 \
 --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu ${BPG} \
 --loss elbo \
 --pl-alpha 0.3 \
 --compile \
 --ckpt-every 5 \
 ${EXTRA} 2>&1

PY_EXIT=$?
fi # end of init-from exists check

echo "Python exit: ${PY_EXIT:-0}"
echo "Finished: $(date)"

# -- Auto-resubmit until done.flag --
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-4}

if [ -f "${OUTDIR}/done.flag" ]; then
 echo "[done] done.flag present - NoHidden BACTERIA-ONLY fine-tune complete for this split"

 # Advance NH_BAC_QUEUE if there are more splits to run.
 # NH_BAC_QUEUE format: space-separated split numbers, e.g. "2 3 4 5".
 if [ -n "${NH_BAC_QUEUE}" ]; then
 NEXT=$(echo "${NH_BAC_QUEUE}" | awk '{print $1}')
 REST=$(echo "${NH_BAC_QUEUE}" | cut -d' ' -f2-)
 [ "${REST}" = "${NH_BAC_QUEUE}" ] && REST=""
 NEXT_EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split${NEXT}/model_ho3.pth --train-feather data/COG_bac_train${NEXT}_phylum.feather --val-feather data/COG_bac_val${NEXT}_phylum.feather --outdir gsd_results_higher_order_nohidden_T20_bac_${CURRICULUM}_split${NEXT}"
 echo "Advancing NH_BAC queue (${CURRICULUM}): split=${NEXT} (${REST:-last})"
 sbatch \
 --export=ALL,CURRICULUM=${CURRICULUM},EXTRA="${NEXT_EXTRA}",NH_BAC_QUEUE="${REST}",HO2_EP=${HO2_EP},HO3_EP=${HO3_EP},BPG=${BPG} \
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
 --export=ALL,CURRICULUM=${CURRICULUM},EXTRA="${EXTRA}",NH_BAC_QUEUE="${NH_BAC_QUEUE}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS},HO2_EP=${HO2_EP},HO3_EP=${HO3_EP},BPG=${BPG} \
 "$0"
fi
