#!/bin/bash
#SBATCH --job-name=gsd-ezoom
#SBATCH --output=gsd-ezoom-%j.out
#SBATCH --error=gsd-ezoom-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --requeue
#
# "Zoom in on E. coli" progressive leave-clade-out fine-tune (research; off-Zenodo).
# Four rungs, IDENTICAL recipe, differing ONLY in which clade around E. coli (the
# Figure 2 genome) is held out of training:
# RANK = phylum | class | order | family
# As the held-out clade shrinks, progressively closer relatives of E. coli enter
# the training set -- the model zooms in. E. coli itself is never trained on (its
# clade is always the held-out one) and is scored separately by reconstruct_extant.
#
# Full HARD retune of the marginal-FP HQ regime with J UNFROZEN (--j-lr-frac 0.1)
# so the couplings absorb the added close relatives (HO2=20 @2e-5, HO3=12 @1e-5).
# Init = the SAME source the published marginal-HQ split-5 used
# (bac_fp_real_hq_split5/model_ho3.pth). Feathers built by
# scripts/build_ecoli_zoom_feathers.py.
#
# Resume-safe (skips done splits; trainer resumes from progress.json on requeue);
# per-rung inductor cache so concurrent --compile jobs do not race; no destructive
# defaults; ASCII only.
#
# Launch all four (2 an 80 GB GPU each -> 8 = the assoc cap):
# for R in phylum class order family; do
# sbatch --gres=gpu:2 --export=ALL,RANK=$R slurm/run_ecoli_zoom_holdout.sh
# done
set -uo pipefail
cd $PROJECT_ROOT

RANK=${RANK:?set RANK=phylum|class|order|family}
BPG=${BPG:-32}
INIT=gsd_results_higher_order_nohidden_T20_bac_fp_real_hq_split5/model_ho3.pth
TF=data/COG_bac_hq_ecolizoom_${RANK}_train.feather
VF=data/COG_bac_hq_ecolizoom_${RANK}_val.feather
OUTDIR=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_ecolizoom_${RANK}

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
echo "=========================================="
echo "E.COLI-ZOOM rung=${RANK} Job ${SLURM_JOB_ID:-?} $(hostname) ${NGPUS} GPUs $(date)"
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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache_ezoom_${RANK}
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

# Pre-flight: init + feathers must exist (else exit cleanly so the resubmit
# chain retries -- e.g. feathers still being built).
MISSING=0
for f in "${INIT}" "${TF}" "${VF}"; do
 if [ ! -f "$f" ]; then
 echo "[wait] required input ${f} not found yet -- will resubmit and retry"
 MISSING=1
 fi
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

# -- Auto-resubmit until done.flag (resume-safe). --
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
