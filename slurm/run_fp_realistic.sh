#!/bin/bash
#SBATCH --job-name=gsd-fp-real
#SBATCH --output=gsd-fp-real-%j.out
#SBATCH --error=gsd-fp-real-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --requeue
#
# RESEARCH RETRAIN -- realistic (coherent) false-positive curriculum.  Family
# names below are NOT in scripts/archive_to_zenodo.sh HO_FAMILIES, so the Zenodo
# deposit never picks them up until we promote them.
#
# WHY: the production FP-tolerant models (mix-FT-fp, bac-FT-fp) were trained with
# INDEPENDENT per-COG random false positives (--fp-max curriculum).  A threshold
# sweep showed that buys almost nothing for the bacterial specialist beyond a
# re-thresholded default (it slides along the precision/recall frontier rather
# than expanding it), because random absent genes rarely fit a genome's
# couplings and are removed easily -- the hard, surviving FPs are functionally
# COHERENT (whole modules, or a related organism's signature).
#
# WHAT: add REALISTIC coherent FP injection on top of the per-COG curriculum --
#   --mod-inject-rate  coherent FP: inject a whole absent functional module
#   --mod-swap-rate    cross-organism FP: graft up to --mod-swap-max modules
#                      from a RANDOM OTHER training genome (a foreign signature
#                      in an incompatible context -- the real reconciliation
#                      failure mode: lateral transfer / mis-mapping in blocks)
# so the model learns to reject the coherent FPs that actually survive.
#
# Two independent arms, 4 GPUs each (8 total, within the assoc cap); each
# arm self-chains its 10 whole-phylum splits via SPLIT_QUEUE.  Schedule + init
# mirror the existing FP fine-tunes (HO2 5 ep @5e-6, HO3 3 ep @1e-6, J frozen,
# beta_hard FN, fp_max=0.1 beta_low) so the ONLY change is the coherent injection.
#   ARM=mix  init mix-FT (T20_mix)        mix feathers -> ..._mix_fp_real_split{N}
#   ARM=bac  init bac-FT (T20_bac_hard)   bac feathers -> ..._bac_fp_real_split{N}
#
# Launch BOTH arms (split 1; chain 2..10), 8 GPUs all out:
#   for A in mix bac; do
#     sbatch --export=ALL,ARM=$A,SPLIT=1,SPLIT_QUEUE="2 3 4 5 6 7 8 9 10" \
#            slurm/run_fp_realistic.sh
#   done

set -uo pipefail
cd $PROJECT_ROOT

ARM=${ARM:?set ARM=mix|bac}
SPLIT=${SPLIT:?set SPLIT (1..10)}
BPG=${BPG:-32}
MODFLAGS="--mod-inject-rate 0.2 --mod-swap-rate 0.3 --mod-swap-max 3"

case "${ARM}" in
  mix) INIT=gsd_results_higher_order_nohidden_T20_mix_split${SPLIT}/model_ho3.pth
       TF=data/COG_mix_train${SPLIT}_phylum.feather
       VF=data/COG_mix_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_higher_order_nohidden_T20_mix_fp_real_split${SPLIT} ;;
  bac) INIT=gsd_results_higher_order_nohidden_T20_bac_hard_split${SPLIT}/model_ho3.pth
       TF=data/COG_bac_train${SPLIT}_phylum.feather
       VF=data/COG_bac_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_higher_order_nohidden_T20_bac_fp_real_split${SPLIT} ;;
  *)   echo "bad ARM=${ARM}"; exit 2 ;;
esac

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
echo "=========================================="
echo "FP-REALISTIC ARM=${ARM} split=${SPLIT} BPG=${BPG}"
echo "Job ${SLURM_JOB_ID:-?} - $(hostname) - ${NGPUS} GPUs - $(date)"
echo "init=${INIT}  outdir=${OUTDIR}"
echo "mod=${MODFLAGS}"
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

if [ ! -f "${INIT}" ]; then
    echo "[wait] init-from ${INIT} not found yet - will resubmit and retry"
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

# -- Auto-resubmit until done.flag; chain SPLIT_QUEUE within the arm. --
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
