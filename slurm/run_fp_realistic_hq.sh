#!/bin/bash
#SBATCH --job-name=gsd-fp-hq
#SBATCH --output=gsd-fp-hq-%j.out
#SBATCH --error=gsd-fp-hq-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --requeue
#
# RESEARCH RETRAIN -- HIGH-COMPLETENESS target + high FN + hard (coherent) FP.
# Off-Zenodo families (mix_fp_real_hq / bac_fp_real_hq) until validated.
#
# Hypothesis: incomplete training genomes inject false-negative LABEL NOISE into
# the target (unassembled-but-present genes scored "absent"), biasing the model
# to under-call.  Fine-tune on near-complete genomes (clean targets) under the
# deep-ancestral noise regime -- high FN (beta_hard) + the coherent realistic FP
# curriculum (whole modules + a foreign organism's grafted module signature).
#
# Data: HQ feathers from scripts/build_hq_feathers.py (per-domain CheckM cut --
# bacteria >=90, archaea >=80, contam <=5; archaea are MAG-dominated so a uniform
# 90% cut would lose 2/3 of them + 25 phyla).  Schedule/init mirror the existing
# FP fine-tune so the deltas are exactly {HQ targets, coherent FP}.
#   ARM=mix  init mix-FT  -> ..._mix_fp_real_hq_split{N}  (COG_mix_hq feathers)
#   ARM=bac  init bac-FT  -> ..._bac_fp_real_hq_split{N}  (COG_bac_hq feathers)
#
# Launch BOTH arms (split 1; chain 2..10), 8 GPUs all out:
#   for A in mix bac; do
#     sbatch --export=ALL,ARM=$A,SPLIT=1,SPLIT_QUEUE="2 3 4 5 6 7 8 9 10" \
#            slurm/run_fp_realistic_hq.sh
#   done

set -uo pipefail
cd $PROJECT_ROOT

ARM=${ARM:?set ARM=mix|bac}
SPLIT=${SPLIT:?set SPLIT (1..10)}
BPG=${BPG:-32}
MODFLAGS="--mod-inject-rate 0.2 --mod-swap-rate 0.3 --mod-swap-max 3"

case "${ARM}" in
  mix) INIT=gsd_results_higher_order_nohidden_T20_mix_split${SPLIT}/model_ho3.pth
       TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
       VF=data/COG_mix_hq_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_higher_order_nohidden_T20_mix_fp_real_hq_split${SPLIT} ;;
  bac) INIT=gsd_results_higher_order_nohidden_T20_bac_hard_split${SPLIT}/model_ho3.pth
       TF=data/COG_bac_hq_train${SPLIT}_phylum.feather
       VF=data/COG_bac_hq_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_higher_order_nohidden_T20_bac_fp_real_hq_split${SPLIT} ;;
  *)   echo "bad ARM=${ARM}"; exit 2 ;;
esac

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
echo "=========================================="
echo "FP-REALISTIC-HQ ARM=${ARM} split=${SPLIT} BPG=${BPG}"
echo "Job ${SLURM_JOB_ID:-?} - $(hostname) - ${NGPUS} GPUs - $(date)"
echo "init=${INIT}  outdir=${OUTDIR}"
echo "feathers=${TF} / ${VF}   mod=${MODFLAGS}"
echo "=========================================="

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo "venv python: $(command -v python3.11)"
else
    echo "WARNING: .venv missing -- module load python/3.11.11"
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    module load python/3.11.11
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

if [ ! -f "${INIT}" ] || [ ! -f "${TF}" ]; then
    echo "[wait] init ${INIT} or feather ${TF} not found yet - will resubmit and retry"
    [ ! -f ${OUTDIR}/progress.json ] && echo '{"completed":[],"waiting_for_checkpoint":true}' > ${OUTDIR}/progress.json
    PY_EXIT=1
else
    python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
        train_higher_order_nohidden.py \
        --init-from      "${INIT}" \
        --new-T          20 \
        --train-feather  "${TF}" --val-feather "${VF}" \
        --module-matrix  data/module_matrix_kegg.pt \
        --ho1-epochs 0 \
        --ho2-epochs 5 --ho2-lr 5e-6 \
        --ho3-epochs 3 --ho3-lr 1e-6 \
        --j-lr-frac      0.0 \
        --ho-fn-curriculum beta_hard \
        --fp-max 0.1 --fp-dist beta_low \
        ${MODFLAGS} \
        --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu ${BPG} \
        --outdir "${OUTDIR}" \
        --loss elbo --pl-alpha 0.3 --compile --ckpt-every 5 \
        2>&1
    PY_EXIT=$?
fi
echo "Python exit: ${PY_EXIT:-0}  -  $(date)"

RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-8}
if [ -f "${OUTDIR}/done.flag" ]; then
    echo "[done] ${OUTDIR} complete"
    if [ -n "${SPLIT_QUEUE:-}" ]; then
        NEXT=$(echo "${SPLIT_QUEUE}" | awk '{print $1}')
        REST=$(echo "${SPLIT_QUEUE}" | cut -d' ' -f2-)
        [ "${REST}" = "${SPLIT_QUEUE}" ] && REST=""
        echo "Advancing ARM=${ARM} -> split=${NEXT} (rest: ${REST:-last})"
        sbatch --export=ALL,ARM=${ARM},SPLIT=${NEXT},SPLIT_QUEUE="${REST}",BPG=${BPG} "$0"
    fi
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
    echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up."
elif [ "${PY_EXIT}" -ne 0 ] && [ ! -f "${OUTDIR}/progress.json" ]; then
    echo "Python exited ${PY_EXIT} with no progress.json - config error, not resubmitting."
else
    NEW_COUNT=$((RESUBMIT_COUNT + 1))
    echo "done.flag missing - resubmitting (${NEW_COUNT}/${MAX_RESUBMITS})"
    sbatch --export=ALL,ARM=${ARM},SPLIT=${SPLIT},SPLIT_QUEUE="${SPLIT_QUEUE:-}",BPG=${BPG},RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS} "$0"
fi
