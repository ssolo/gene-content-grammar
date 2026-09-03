#!/bin/bash
#SBATCH --job-name=gsd-ah-marg
#SBATCH --output=gsd-ah-marg-%j.out
#SBATCH --error=gsd-ah-marg-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --requeue
#
# ADD-HIDDEN MARGINAL -- bolt ~H transparent hidden spins onto the TRAINED marginal
# (the trimmer) and continue the marginal-FP PRUNE curriculum under the TIED ELBO
# (onsager=tied, U=A^T -> clean single-matrix EBM, proper variational bound).
#
# Key vs the failed strong-marginal run: init from the MARGINAL (already trims),
# NOT mix_fp_real_hq (rescue-leaning).  train_add_hidden zero-inits A (transparent:
# output == source until gradients engage), warms the hidden up with a boosted
# module-aux loss (module coherence), then we add the prune curriculum so the
# hidden also learn the higher-order TRIM.  Broad prior by default (no
# --marginal-freq-feather); set MARGFREQ for a domain/blended prior.
#
#   ARM=mix init mix-FT-fp-marginal-hq -> gsd_results_addhidden_T20_mix_fp_marginal_ah_hq_split{N}
#   ARM=bac init bac-FT-fp-marginal-hq -> gsd_results_addhidden_T20_bac_fp_marginal_ah_hq_split{N}
# PoC:  sbatch --export=ALL,ARM=mix,SPLIT=1 slurm/run_add_hidden_marginal.sh
# Fan:  add SPLIT_QUEUE="2 3"
# After: python scripts/validate_marginal.py --model mix-FT-fp-marginal-ah-hq --splits 1-3 --gld laca

set -uo pipefail
cd $PROJECT_ROOT

ARM=${ARM:?set ARM=mix|bac}
SPLIT=${SPLIT:?set SPLIT (1..10)}
H=${H:-100}
K=${K:-6}
BPG=${BPG:-32}
AH1=${AH1:-12}; AH2=${AH2:-15}; AH3=${AH3:-8}   # epoch schedule (shorten for a quick read)
# Marginal-FP PRUNE curriculum (the trim signal); broad prior unless MARGFREQ set.
MARGFLAGS="--marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15"
MARGFREQ=${MARGFREQ:-}   # e.g. "--marginal-freq-feather data/COG_arc_train${SPLIT}_phylum.feather"

case "${ARM}" in
  mix) INIT=gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
       VF=data/COG_mix_hq_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_addhidden_T20_mix_fp_marginal_ah_hq_split${SPLIT} ;;
  bac) INIT=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_bac_hq_train${SPLIT}_phylum.feather
       VF=data/COG_bac_hq_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_addhidden_T20_bac_fp_marginal_ah_hq_split${SPLIT} ;;
  *)   echo "bad ARM=${ARM}"; exit 2 ;;
esac

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
echo "=========================================="
echo "ADD-HIDDEN MARGINAL ARM=${ARM} split=${SPLIT} H=${H} K=${K} onsager=tied"
echo "Job ${SLURM_JOB_ID:-?} - $(hostname) - ${NGPUS} GPUs - $(date)"
echo "init=${INIT}  outdir=${OUTDIR}"
echo "marg=${MARGFLAGS} ${MARGFREQ:-(broad prior)}"
echo "=========================================="

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
    module load python/3.11.11
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

if [ ! -f "${INIT}" ]; then
    echo "[wait] init-from ${INIT} not found yet - will resubmit and retry"
    [ ! -f ${OUTDIR}/progress.json ] && echo '{"completed":[],"waiting_for_checkpoint":true}' > ${OUTDIR}/progress.json
    PY_EXIT=1
else
    python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
        train_add_hidden.py \
        --init-from      "${INIT}" \
        --new-T          20 \
        --train-feather  "${TF}" --val-feather "${VF}" \
        --module-matrix  data/module_matrix_kegg.pt \
        --H ${H} --K ${K} --onsager tied --loss elbo --pl-alpha 0.3 \
        --ah1-epochs ${AH1} --ah2-epochs ${AH2} --ah3-epochs ${AH3} \
        --j-lr-frac      0.0 \
        ${MARGFLAGS} ${MARGFREQ} \
        --batch-per-gpu ${BPG} --ms-steps 3 \
        --outdir "${OUTDIR}" --compile --ckpt-every 5 \
        2>&1
    PY_EXIT=$?
fi
echo "Python exit: ${PY_EXIT:-0}  -  $(date)"

# -- Auto-resubmit until done.flag; chain SPLIT_QUEUE within the arm (resume-safe). --
RESUBMIT_COUNT=${RESUBMIT_COUNT:-0}
MAX_RESUBMITS=${MAX_RESUBMITS:-8}
if [ -f "${OUTDIR}/done.flag" ]; then
    echo "[done] ${OUTDIR} complete"
    if [ -n "${SPLIT_QUEUE:-}" ]; then
        NEXT=$(echo "${SPLIT_QUEUE}" | awk '{print $1}')
        REST=$(echo "${SPLIT_QUEUE}" | cut -d' ' -f2-)
        [ "${REST}" = "${SPLIT_QUEUE}" ] && REST=""
        echo "Advancing ARM=${ARM} -> split=${NEXT} (rest: ${REST:-last})"
        sbatch --export=ALL,ARM=${ARM},SPLIT=${NEXT},SPLIT_QUEUE="${REST}",H=${H},K=${K},BPG=${BPG},AH1=${AH1},AH2=${AH2},AH3=${AH3},MARGFREQ="${MARGFREQ}" "$0"
    fi
elif [ "${RESUBMIT_COUNT}" -ge "${MAX_RESUBMITS}" ]; then
    echo "MAX_RESUBMITS=${MAX_RESUBMITS} reached - giving up."
elif [ "${PY_EXIT}" -ne 0 ] && [ ! -f "${OUTDIR}/progress.json" ]; then
    echo "Python exited ${PY_EXIT} with no progress.json - config error, not resubmitting."
else
    NEW_COUNT=$((RESUBMIT_COUNT + 1))
    echo "done.flag missing - resubmitting (${NEW_COUNT}/${MAX_RESUBMITS})"
    sbatch --export=ALL,ARM=${ARM},SPLIT=${SPLIT},SPLIT_QUEUE="${SPLIT_QUEUE:-}",H=${H},K=${K},BPG=${BPG},AH1=${AH1},AH2=${AH2},AH3=${AH3},MARGFREQ="${MARGFREQ}",RESUBMIT_COUNT=${NEW_COUNT},MAX_RESUBMITS=${MAX_RESUBMITS} "$0"
fi
