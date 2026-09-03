#!/bin/bash
#SBATCH --job-name=gsd-ho-nohidden
#SBATCH --output=gsd-ho-nohidden-%j.out
#SBATCH --error=gsd-ho-nohidden-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
# 4 GPUs (not 8): runs two concurrent chains under the 8-GPU assoc cap.
# Total wall time is bottlenecked by the cap, not per-job count.
# Memory stays at 240G (host memory, not GPU); per-GPU BPG=256 unchanged.
#
# SLURM script for the cluster (GPU nodes, 80 GB).
# Add visible-only attention Delta to a NoHiddenDenoiser checkpoint.
# No hidden spins, no auxiliary loss.
#
# Usage:
# sbatch --export=ALL,EXTRA="--init-from <ckpt.pth> --outdir <out>" \
# slurm/run_higher_order_nohidden.sh 12
#
# T=12 sweep (from finetune chain f3 checkpoints):
# for SPLIT in {1..10}; do
# sbatch --export=ALL,EXTRA="\
# --init-from gsd_results_nohidden_finetune_chain_T8to20_split${SPLIT}/model_T8to12_f3.pth \
# --train-feather data/COG_train${SPLIT}_phylum.feather \
# --val-feather data/COG_val${SPLIT}_phylum.feather \
# --outdir gsd_results_higher_order_nohidden_T12_split${SPLIT}" \
# slurm/run_higher_order_nohidden.sh 12
# done
#
# T=16 sweep:
# for SPLIT in {1..10}; do
# sbatch --export=ALL,EXTRA="\
# --init-from gsd_results_nohidden_finetune_chain_T8to20_split${SPLIT}/model_T12to16_f3.pth \
# --train-feather data/COG_train${SPLIT}_phylum.feather \
# --val-feather data/COG_val${SPLIT}_phylum.feather \
# --outdir gsd_results_higher_order_nohidden_T16_split${SPLIT}" \
# slurm/run_higher_order_nohidden.sh 16
# done

T=${1:-12}
NGPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "=========================================="
echo "HO NoHidden T=${T} - Job $SLURM_JOB_ID - $(hostname) - ${NGPUS} GPUs"
echo "Started: $(date)"
echo "=========================================="

cd $PROJECT_ROOT

# Prefer the project venv: its python3.11 is a symlink into the
# cluster's module python install on shared FS, so it works even when
# 'module load python/3.11.11' fails because some unrelated Lmod
# dependency (e.g. openmpi.gcc/5.0.3) is missing on the node we
# landed on (a GPU node has been hitting that on the HO chain).
# Only fall back to 'module load' if the venv is absent.
if [ -f .venv/bin/activate ]; then
 source .venv/bin/activate
 echo "Activated venv python: $(command -v python3.11) -> $(readlink -f "$(command -v python3.11)")"
else
 echo "WARNING: .venv missing -- falling back to module load python/3.11.11"
 [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
 module load python/3.11.11
fi

OUTDIR=$(echo "${EXTRA}" | grep -oE -- '--outdir[[:space:]]+[^[:space:]]+' | awk '{print $2}')
OUTDIR=${OUTDIR:-gsd_results_higher_order_nohidden_T${T}}
echo "Outdir: ${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

echo "EXTRA: ${EXTRA}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p ${TORCHINDUCTOR_CACHE_DIR}

# BPG scaled by T: T=12 -> 384, T=16 -> 256.
# NoHidden saves ~10-15% memory vs HO-with-hidden; gradient checkpointing
# on attention is ON by default. Re-profile with run_profile_higher_order.sh
# (with --no-hidden) if you change attention hyperparameters.
# BPG scaled by T to keep memory <= proven empirical baseline.
#
# Empirical data point (profile_higher_order_memory.py, T=12 hidden HO):
# BPG=256 -> 39.5 GiB (50% of 80 GiB).
#
# Activation memory prop to BPG x T, so we keep BPG x T ~ const ~ 3072 (= 256 x 12):
# T=8 BPG=384 ~ 40 GiB
# T=12 BPG=256 ~ 40 GiB (matches baseline; NoHidden may save a bit)
# T=16 BPG=192 ~ 40 GiB
# T=20 BPG=128 ~ 33 GiB (extra safety margin)
#
# Gradient checkpointing on attention is ON. Speed comes from reduced
# epochs (50 vs original 80), not from disabling checkpointing.
if [ "${T}" -le 8 ]; then
 BPG=384
elif [ "${T}" -le 12 ]; then
 BPG=256
elif [ "${T}" -le 16 ]; then
 BPG=192
else
 BPG=128
fi

# -- Pre-flight: check that --init-from checkpoint exists --
# If queued before the finetune chain finishes, the checkpoint won't
# exist yet. Instead of crashing as a "config error" (which blocks
# auto-resubmit), exit cleanly so the resubmit chain retries later.
INIT_FROM=$(echo "${EXTRA}" | grep -oE -- '--init-from[[:space:]]+[^[:space:]]+' | awk '{print $2}')
if [ -n "${INIT_FROM}" ] && [ ! -f "${INIT_FROM}" ]; then
 echo "[wait] --init-from ${INIT_FROM} not found yet - will resubmit and retry"
 mkdir -p ${OUTDIR}
 [ ! -f ${OUTDIR}/progress.json ] && echo '{"completed":[],"waiting_for_checkpoint":true}' > ${OUTDIR}/progress.json
 PY_EXIT=1
else

python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
 train_higher_order_nohidden.py \
 --module-matrix data/module_matrix_kegg.pt \
 --new-T ${T} \
 --ho1-epochs 20 --ho1-lr 1e-4 \
 --ho2-epochs 40 --ho2-lr 5e-5 \
 --ho3-epochs 20 --ho3-lr 2e-5 \
 --j-lr-frac 0.1 \
 --attn-d-model 128 --attn-nhead 4 --attn-n-layers 1 --attn-dim-ff 512 \
 --attn-lr-frac 0.3 \
 --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu ${BPG} \
 --outdir ${OUTDIR} \
 --loss elbo \
 --pl-alpha 0.3 \
 --onsager full \
 --compile \
 --ckpt-every 5 \
 ${EXTRA} 2>&1

PY_EXIT=$?
fi # end of init-from exists check

echo "Python exit: ${PY_EXIT:-0}"
echo "Finished: $(date)"

# -- Auto-resubmit until done.flag --
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-8}

if [ -f "${OUTDIR}/done.flag" ]; then
 echo "[done] done.flag present - training complete"

 # -- Advance HO_QUEUE if there are more (split, T) pairs to run --
 # HO_QUEUE format: "S1:T1 S2:T2 S3:T3 ..." (space-separated).
 # Pop the first item, build EXTRA for it, and submit; pass the
 # remainder forward via HO_QUEUE.
 if [ -n "${HO_QUEUE}" ]; then
 NEXT=$(echo "${HO_QUEUE}" | awk '{print $1}')
 REST=$(echo "${HO_QUEUE}" | cut -d' ' -f2-)
 [ "${REST}" = "${HO_QUEUE}" ] && REST="" # only 1 item left
 NEXT_SPLIT=${NEXT%:*}
 NEXT_T=${NEXT#*:}
 # Init checkpoint differs by T:
 # T=8 -> s3e of the NoHidden denovo training (no extension, no chain)
 # T=12/16/20 -> corresponding f3 leg of the finetune chain
 case ${NEXT_T} in
 8) NEXT_INIT_PATH="gsd_results_nohidden_denovo_elbo_T8_split${NEXT_SPLIT}/model_s3e.pth";;
 12) NEXT_INIT_PATH="gsd_results_nohidden_finetune_chain_T8to20_split${NEXT_SPLIT}/model_T8to12_f3.pth";;
 16) NEXT_INIT_PATH="gsd_results_nohidden_finetune_chain_T8to20_split${NEXT_SPLIT}/model_T12to16_f3.pth";;
 20) NEXT_INIT_PATH="gsd_results_nohidden_finetune_chain_T8to20_split${NEXT_SPLIT}/model_T16to20_f3.pth";;
 *) echo "Unknown NEXT_T=${NEXT_T} - chain ends"; exit 0;;
 esac
 # Epoch counts must match launch_ho_chain.sh: HO1=5, HO2=15, HO3=8.
 # See that file for the May 2026 convergence-analysis rationale.
 NEXT_EXTRA="--init-from ${NEXT_INIT_PATH} --train-feather data/COG_train${NEXT_SPLIT}_phylum.feather --val-feather data/COG_val${NEXT_SPLIT}_phylum.feather --ho1-epochs 5 --ho2-epochs 15 --ho3-epochs 8 --outdir gsd_results_higher_order_nohidden_T${NEXT_T}_split${NEXT_SPLIT}"
 echo "Advancing chain: split=${NEXT_SPLIT} T=${NEXT_T} (${REST:-last})"
 sbatch \
 --export=ALL,EXTRA="${NEXT_EXTRA}",HO_QUEUE="${REST}" \
 "$0" ${NEXT_T}
 fi
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
 echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up."
elif [ "${PY_EXIT}" -ne 0 ] && [ ! -f "${OUTDIR}/progress.json" ]; then
 echo "Python exited ${PY_EXIT} with no progress.json - config error, not resubmitting."
else
 NEW_COUNT=$((RESUBMIT_COUNT + 1))
 echo "done.flag missing - resubmitting (chain ${NEW_COUNT}/${MAX_RESUBMITS})"
 # Preserve HO_QUEUE across retries so chain advances after eventual completion.
 sbatch \
 --export=ALL,EXTRA="${EXTRA}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS},HO_QUEUE="${HO_QUEUE}" \
 "$0" "$@"
fi
