#!/bin/bash
#SBATCH --job-name=gsd-ezoomv
#SBATCH --output=gsd-ezoomv-%j.out
#SBATCH --error=gsd-ezoomv-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --requeue
#
# a 16 GB GPU variant of run_ecoli_zoom_holdout.sh -- runs a single "zoom in on E. coli"
# rung on 4 GPUs (16 GB each, the partition cap), ADDITIVE to the 8 large-memory GPUs used by
# the rungs, so it runs fully concurrently. Used for the closest rung(s):
# RANK = genus (hold out only g__Escherichia; closest legit relatives)
#
# IDENTICAL recipe to the 80 GB GPU launcher (full marginal-FP HQ retune, J UNFROZEN,
# HO2=20@2e-5 HO3=12@1e-5, init bac_fp_real_hq_split5). Only the hardware differs:
# a 16 GB GPU-32GB -> smaller per-GPU batch (BPG=24 default; 4 GPUs keep ~the same global
# batch as 2 an 80 GB GPU @ BPG=32... 96 vs 64, close). Pins gpu: so it never lands a
# useless an older GPU (). Resume-safe; per-rung inductor cache; ASCII only.
#
# sbatch --export=ALL,RANK=genus slurm/run_ecoli_zoom_holdout_smallgpu.sh
set -uo pipefail
cd $PROJECT_ROOT

RANK=${RANK:?set RANK=genus|family|order|class|phylum}
BPG=${BPG:-24}
INIT=gsd_results_higher_order_nohidden_T20_bac_fp_real_hq_split5/model_ho3.pth
TF=data/COG_bac_hq_ecolizoom_${RANK}_train.feather
VF=data/COG_bac_hq_ecolizoom_${RANK}_val.feather
OUTDIR=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_${RANK}

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
echo "=========================================="
echo "E.COLI-ZOOM-a 16 GB GPU rung=${RANK} Job ${SLURM_JOB_ID:-?} $(hostname) ${NGPUS} GPUs $(date)"
echo "init=${INIT}"
echo "train=${TF}"
echo "val=${VF}"
echo "outdir=${OUTDIR}"
echo "=========================================="

if [ -f "${OUTDIR}/done.flag" ]; then echo "[skip] ${OUTDIR} already done"; exit 0; fi

if [ -f .venv/bin/activate ]; then
 source .venv/bin/activate
 echo "venv python: $(command -v python3.11)"
else
 echo "WARNING: .venv missing -- module load python/3.11.11"
 [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
 module load python/3.11.11
fi

# Sanity: a 16 GB GPU (not an older GPU) and CUDA up. an older GPU nodes break torch+pyarrow.
python3.11 -c "import torch;assert torch.cuda.is_available(),'no CUDA';print('GPU',torch.cuda.get_device_name(0))" || {
 echo "[abort] CUDA unavailable (likely a an older GPU node) -- resubmit"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache_ezoom_v100_${RANK}
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

MISSING=0
for f in "${INIT}" "${TF}" "${VF}"; do
 if [ ! -f "$f" ]; then echo "[wait] required input ${f} not found yet -- resubmit+retry"; MISSING=1; fi
done

if [ "${MISSING}" = "1" ]; then
 [ ! -f ${OUTDIR}/progress.json ] && \
 echo '{"completed":[],"waiting_for_input":true}' > ${OUTDIR}/progress.json
 PY_EXIT=1
else
 python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
 train_higher_order_nohidden.py \
 --init-from "${INIT}" \
 --new-T 20 \
 --train-feather "${TF}" --val-feather "${VF}" \
 --module-matrix data/module_matrix_kegg.pt \
 --ho1-epochs 0 \
 --ho2-epochs 20 --ho2-lr 2e-5 \
 --ho3-epochs 12 --ho3-lr 1e-5 \
 --j-lr-frac 0.1 \
 --ho-fn-curriculum beta_hard \
 --fp-max 0.1 --fp-dist beta_low \
 --mod-inject-rate 0.1 --mod-swap-rate 0.2 --mod-swap-max 3 \
 --marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15 \
 --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu ${BPG} \
 --outdir "${OUTDIR}" \
 --loss elbo --pl-alpha 0.3 --compile --ckpt-every 5 \
 2>&1
 PY_EXIT=$?
fi
echo "Python exit: ${PY_EXIT:-0} - $(date)"

RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-6}
if [ -f "${OUTDIR}/done.flag" ]; then
 echo "[done] ${OUTDIR} complete"
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
 echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached -- giving up."
elif [ "${PY_EXIT:-1}" -ne 0 ] && [ ! -f "${OUTDIR}/progress.json" ]; then
 echo "Python exited ${PY_EXIT:-1} with no progress.json -- config error, not resubmitting."
else
 NEW_COUNT=$((RESUBMIT_COUNT + 1))
 echo "done.flag missing -- resubmitting (${NEW_COUNT}/${MAX_RESUBMITS})"
 sbatch --gres=gpu:${NGPUS} \
 --export=ALL,RANK=${RANK},BPG=${BPG},RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS} \
 "$0"
fi
