#!/bin/bash
#SBATCH --job-name=gsd-ho-nh-mix-T20
#SBATCH --output=gsd-ho-nh-mix-T20-%j.out
#SBATCH --error=gsd-ho-nh-mix-T20-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#
# 4 GPUs per split (launcher runs TWO splits concurrently under the # 8-GPU assoc cap, exactly like launch_ho_chain.sh / the other arc + AH
# launchers).
#
# SLURM script for the cluster (GPU nodes, 80 GB).
# Fine-tune a NoHidden HO3 T=20 model on MIXED ARCHAEA+BACTERIA data,
# specialising it toward archaeal patterns WHILE PRESERVING the
# bacterial co-occurrence regulariser via a bacterial sub-sample in
# every minibatch. The fix for the archaea-only fine-tune's
# catastrophic forgetting: only HO2 + HO3 with reduced epochs, small
# LRs, and J frozen.
#
#
# DIFFERENCES vs run_higher_order_nohidden_T20_arc_finetune.sh:
# - data: data/COG_mix_{train,val}{N}_phylum.feather (build via
# scripts/build_archaeal_feathers.py --mix-bacteria-ratio 1.0)
# - schedule: HO2=5 ep@5e-6, HO3=3 ep@1e-6 (was 10 ep@1e-5, 10 ep@5e-6)
# -- 8x less drift than the arc-only fine-tune, which regressed
# - --j-lr-frac 0.0 (J FROZEN; arc had 0.1)
# - --aux-lambda 0.0 (unchanged; NoHidden has no hidden-state aux head)
# - outputs: gsd_results_higher_order_nohidden_T20_mix_split{N}/
#
# Source : gsd_results_higher_order_nohidden_T20_split{1..10}/model_ho3.pth
# Target : gsd_results_higher_order_nohidden_T20_mix_split{1..10}/model_ho3.pth
#
# This is the no-hidden / T=20 counterpart of run_higher_order_arc_finetune.sh
# (which targets HO_tied T=8). Key differences vs that script:
# - uses train_higher_order_nohidden.py (the NoHidden HO trainer that
# produced the source; correctly handles a NoHidden + attention source)
# - --new-T 20 (vs --new-T 8)
# - --aux-lambda 0.0 (NoHiddenDenoiser has no hidden state -> no
# module-completeness aux loss; the trainer accepts 0)
# - --onsager OMITTED so the trainer auto-resolves from the source
# checkpoint's .cfg.json sidecar (NoHidden uses boolean onsager,
# not the HigherOrderDenoiser 'full'/'tied' modes)
# - 4 GPUs/split (vs 8) so two splits can run concurrently under the
# 8-GPU cap, matching the AddHidden + FP-finetune launchers
#
# Schedule (small-drift defaults; J FROZEN via --j-lr-frac 0.0; ~8x less
# total drift than the arc-only fine-tune, which regressed):
# HO1: 0 ep (skipped; attention already warm)
# HO2: 5 ep at 5e-6
# HO3: 3 ep at 1e-6
# BPG: 32 (vs the joint T=20 NoHidden's 128 -- ~10k mixed genomes per
# split (5k archaea + 5k bacteria at ratio=1.0), so smaller
# BPG keeps the per-epoch gradient-update count meaningful;
# ~80 updates/epoch on 4 GPUs vs ~20 at BPG=128)
#
# Wall-time estimate per split: ~25-40 min. Twice the data of the
# arc-only fine-tune (~10k vs ~5k genomes/split), but only 8 epochs
# total (vs 20 in the original arc ft), so ~similar wall. 10 splits
# across 2 concurrent chains -> ~2.5-3.5 h wall.
#
# Usage (single split):
# sbatch --export=ALL,EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth \
# --train-feather data/COG_mix_train1_phylum.feather \
# --val-feather data/COG_mix_val1_phylum.feather \
# --outdir gsd_results_higher_order_nohidden_T20_mix_split1" \
# slurm/run_higher_order_nohidden_T20_mix_finetune.sh
#
# Normally launched via scripts/launch_ho_nohidden_T20_mix_finetune.sh.

echo "=========================================="
echo "NoHidden HO T=20 MIXED-DOMAIN fine-tune - Job $SLURM_JOB_ID - $(hostname)"
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

OUTDIR=$(echo "${EXTRA}" | grep -oE -- '--outdir[[:space:]]+[^[:space:]]+' | awk '{print $2}')
OUTDIR=${OUTDIR:-gsd_results_higher_order_nohidden_T20_mix}
echo "Outdir: ${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"
echo "EXTRA: ${EXTRA}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

# BPG=32 (override-able): with ~10k mixed genomes per split (5k archaea
# + 5k bacteria at --mix-bacteria-ratio 1.0) and 4 GPUs, BPG=32 gives
# ~60-80 updates/epoch. At BPG=128 (the joint default) it would be
# ~20/epoch -- still workable, but at our 8-epoch total (HO2=5 + HO3=3)
# we want enough updates per epoch for the small-LR drift to be smooth.
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
 echo "[done] done.flag present - NoHidden MIXED fine-tune complete for this split"

 # Advance NH_MIX_QUEUE if there are more splits to run.
 # NH_MIX_QUEUE format: space-separated split numbers, e.g. "2 3 4 5".
 if [ -n "${NH_MIX_QUEUE}" ]; then
 NEXT=$(echo "${NH_MIX_QUEUE}" | awk '{print $1}')
 REST=$(echo "${NH_MIX_QUEUE}" | cut -d' ' -f2-)
 [ "${REST}" = "${NH_MIX_QUEUE}" ] && REST=""
 NEXT_EXTRA="--init-from gsd_results_higher_order_nohidden_T20_split${NEXT}/model_ho3.pth --train-feather data/COG_mix_train${NEXT}_phylum.feather --val-feather data/COG_mix_val${NEXT}_phylum.feather --outdir gsd_results_higher_order_nohidden_T20_mix_split${NEXT}"
 echo "Advancing NH_MIX queue: split=${NEXT} (${REST:-last})"
 sbatch \
 --export=ALL,EXTRA="${NEXT_EXTRA}",NH_MIX_QUEUE="${REST}",HO2_EP=${HO2_EP},HO3_EP=${HO3_EP},BPG=${BPG} \
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
 --export=ALL,EXTRA="${EXTRA}",NH_MIX_QUEUE="${NH_MIX_QUEUE}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS},HO2_EP=${HO2_EP},HO3_EP=${HO3_EP},BPG=${BPG} \
 "$0"
fi
