#!/bin/bash
#SBATCH --job-name=gsd-fp-marg
#SBATCH --output=gsd-fp-marg-%j.out
#SBATCH --error=gsd-fp-marg-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=240G
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --requeue
#
# RESEARCH RETRAIN -- MARGINAL-FP curriculum (balanced).  Family names
# ..._{arm}_fp_marginal_split{N} are NOT in archive_to_zenodo.sh.
#
# Goal: decide whether the high-count softlanding LBCA *input* (2014 genes but
# only 86 complete KEGG modules -- broad but module-INCOHERENT, the Count/
# Brownian per-family-marginal signature, NOT a module-complete consensus) is
# over-reconstruction the model should trim.
#
# Each genome's K copies are split (same clean target throughout):
#   - clean copies (nc): preserve real genomes.
#   - RESCUE copies: high FN (beta_hard) + low FP (fp_max 0.1) [+ light coherent
#     FP] -> the model KEEPS its fill-in ability (sparse reconciliation inputs).
#   - MARGINAL-PRUNE copies: low FN + dense FP drawn by per-COG cross-genome
#     marginal frequency (--marginal-fp-*) -> a broad, low-co-occurrence-
#     coherence over-reconstruction the model must TRIM back to the clean genome.
# The model reads off the input's coupling coherence whether to fill in or trim.
# Ceiling: only coupling-VIOLATING excess is trimmable; generically-plausible
# genes can't be called contamination by coherence alone.
#
# Stick to HQ: init from the HQ realistic-FP production model and train on HQ
# (>=90% CheckM) feathers, so the clean PRUNE target is a genuinely COMPLETE
# genome (else the model would learn to trim toward incomplete genomes) and the
# marginals reflect complete-genome frequencies; also directly comparable to the
# focal LBCA, which uses bac-FT-fp-HQ.
#   ARM=bac  init bac-FT-fp-HQ (bac_fp_real_hq), HQ feathers -> ..._bac_fp_marginal_hq_split{N}
#   ARM=mix  init mix-FT-fp-HQ (mix_fp_real_hq), HQ feathers -> ..._mix_fp_marginal_hq_split{N}
#   ARM=marc init mix-FT-fp-HQ, mix HQ feathers, contamination prior = ARCHAEA p_c
#            (COG_arc_train) -> ..._mix_fp_margarc_hq_split{N}
#   ARM=marbac init mix-FT-fp-HQ, mix HQ feathers, contamination prior = BACTERIA p_c
#            (COG_bac_train) -> ..._mix_fp_margbac_hq_split{N}
# PoC (bac, split 1):  sbatch --export=ALL,ARM=bac,SPLIT=1 slurm/run_fp_marginal.sh
# Full arm:            add SPLIT_QUEUE="2 3 4 5 6 7 8 9 10"

set -uo pipefail
cd $PROJECT_ROOT

ARM=${ARM:?set ARM=mix|bac}
SPLIT=${SPLIT:?set SPLIT (1..10)}
BPG=${BPG:-32}
# light coherent FP on rescue copies + the balanced marginal-prune curriculum:
MODFLAGS="--mod-inject-rate 0.1 --mod-swap-rate 0.2 --mod-swap-max 3"
MARGFLAGS="--marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15"
MARGFREQ=""   # optional external contamination prior (set per-arm below)

case "${ARM}" in
  mix) INIT=gsd_results_higher_order_nohidden_T20_mix_fp_real_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
       VF=data/COG_mix_hq_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split${SPLIT} ;;
  bac) INIT=gsd_results_higher_order_nohidden_T20_bac_fp_real_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_bac_hq_train${SPLIT}_phylum.feather
       VF=data/COG_bac_hq_val${SPLIT}_phylum.feather
       OUTDIR=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split${SPLIT} ;;
  marc) INIT=gsd_results_higher_order_nohidden_T20_mix_fp_real_hq_split${SPLIT}/model_ho3.pth
        TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
        VF=data/COG_mix_hq_val${SPLIT}_phylum.feather
        MARGFREQ="--marginal-freq-feather data/COG_arc_train${SPLIT}_phylum.feather"
        OUTDIR=gsd_results_higher_order_nohidden_T20_mix_fp_margarc_hq_split${SPLIT} ;;
  marbac) INIT=gsd_results_higher_order_nohidden_T20_mix_fp_real_hq_split${SPLIT}/model_ho3.pth
        TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
        VF=data/COG_mix_hq_val${SPLIT}_phylum.feather
        MARGFREQ="--marginal-freq-feather data/COG_bac_train${SPLIT}_phylum.feather"
        OUTDIR=gsd_results_higher_order_nohidden_T20_mix_fp_margbac_hq_split${SPLIT} ;;
  *)   echo "bad ARM=${ARM}"; exit 2 ;;
esac

NGPUS=$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)
echo "=========================================="
echo "FP-MARGINAL ARM=${ARM} split=${SPLIT} BPG=${BPG}"
echo "Job ${SLURM_JOB_ID:-?} - $(hostname) - ${NGPUS} GPUs - $(date)"
echo "init=${INIT}  outdir=${OUTDIR}"
echo "mod=${MODFLAGS}  marg=${MARGFLAGS}"
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
        --ho2-epochs 6 --ho2-lr 8e-6 \
        --ho3-epochs 4 --ho3-lr 2e-6 \
        --j-lr-frac      0.0 \
        --ho-fn-curriculum beta_hard \
        --fp-max 0.1 --fp-dist beta_low \
        ${MODFLAGS} ${MARGFLAGS} ${MARGFREQ} \
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
    if [ -z "${SPLIT_QUEUE:-}" ]; then
        # last split of this arm done -> chain its spectra eval (cluster-side).
        case "${ARM}" in
          marc)   sbatch --export=ALL,MODEL=margarc,VAL=COG_arc_val,VALTAG=_arcval --array=1-3%3 slurm/run_marginal_spectra.sbatch && echo "[chain] margarc spectra (arc holdout) submitted" ;;
          marbac) sbatch --export=ALL,MODEL=margbac,VAL=COG_bac_val,VALTAG=_bacval --array=1-3%3 slurm/run_marginal_spectra.sbatch && echo "[chain] margbac spectra (bac holdout) submitted" ;;
        esac
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
