#!/bin/bash
#SBATCH --job-name=gsd-cons
#SBATCH --output=gsd-cons-%A_%a.out
#SBATCH --error=gsd-cons-%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#
# CROSS-INPUT CONSISTENCY quick test.
# Fine-tune the TRAINED marginal (the trimmer) with ONLY the cross-input
# consistency loss added on top of the existing marginal-FP curriculum:
# penalise the variance across the K corrupted copies of each genome so the
# sparse/under- and dense/over-corrupted copies map to the SAME denoised output
# (input-invariance).  Hypothesis: this collapses the sparse-build (~1018) vs
# dense-trim (~1767) fixed-point gap that neither strong-marginal nor add-hidden
# touched.
#
# "everything open": init from the marginal, skip ho1/ho2, run ho3 (full joint,
# J fully trainable via --j-lr-frac 1.0) for a few epochs.  No hidden units.
#
#   sbatch --export=ALL,ARM=mix,LAMBDA=1.0 --array=1 slurm/run_consistency_test.sh
#   sbatch --export=ALL,ARM=mix,LAMBDA=3.0 --array=1 slurm/run_consistency_test.sh
#   bac arm: ARM=bac ;  more splits: --array=1-4
# Validate (gap test) runs inline at the end; also re-derivable with:
#   python scripts/analyze_ancestral_node.py --table LACA_GLD_min1_input.tsv --node LACA_GLD_min1 \
#          --models "<outdir>/model_ho3.pth" --actual-mode raw --csv-out /tmp/x.tsv --output /dev/null
set -uo pipefail
cd $PROJECT_ROOT
ARM=${ARM:?set ARM=mix|bac}
SPLIT=${SLURM_ARRAY_TASK_ID:?launch with --array=1-N}
K=${K:-6}; BPG=${BPG:-32}; LAMBDA=${LAMBDA:-1.0}
# Recipe: FAITHFUL to the original marginal by default -- identical schedule and
# gentle LRs as run_fp_marginal_array.sh (ho2 6 @ 8e-6 + ho3 4 @ 2e-6), J FROZEN
# (j-lr-frac 0.0). The ONLY change vs the marginal is the added consistency term,
# so any gap change is attributable to it. Open-J variant: JLR=1.0 HO2=0 HO3=6.
HO2=${HO2:-6}; HO2LR=${HO2LR:-8e-6}; HO3=${HO3:-4}; HO3LR=${HO3LR:-2e-6}; JLR=${JLR:-0.0}
MARGFLAGS="--marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15"

case "${ARM}" in
  mix) INIT=gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
       VF=data/COG_mix_hq_val${SPLIT}_phylum.feather ;;
  bac) INIT=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_bac_hq_train${SPLIT}_phylum.feather
       VF=data/COG_bac_hq_val${SPLIT}_phylum.feather ;;
  *)   echo "bad ARM=${ARM}"; exit 2 ;;
esac
OUTDIR=gsd_results_consistency_T20_${ARM}_fp_marginal_cons_l${LAMBDA}_j${JLR}_hq_split${SPLIT}

echo "CONSISTENCY ARM=${ARM} split=${SPLIT} K=${K} lambda=${LAMBDA} ho2=${HO2}@${HO2LR} ho3=${HO3}@${HO3LR} j-lr-frac=${JLR}  $(hostname) $(date)"
echo "init=${INIT} -> outdir=${OUTDIR}  marg=${MARGFLAGS}"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; module load python/3.11.11; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache_cons_s${SPLIT}
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"

[ -f "${OUTDIR}/done.flag" ] && { echo "[skip] ${OUTDIR} done"; exit 0; }
[ -s "${INIT}" ] || { echo "[no-init] ${INIT} not found"; exit 1; }
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

# PLAIN python => single GPU, single process => no DDP.
# ho1/ho2 epochs 0 -> stages skipped (init-from still loads); ho3 only.
python3.11 train_higher_order_nohidden.py \
    --init-from "${INIT}" --new-T 20 \
    --train-feather "${TF}" --val-feather "${VF}" \
    --module-matrix data/module_matrix_kegg.pt \
    --K ${K} --loss elbo --pl-alpha 0.3 \
    --ho1-epochs 0 --ho2-epochs ${HO2} --ho2-lr ${HO2LR} --ho3-epochs ${HO3} --ho3-lr ${HO3LR} --j-lr-frac ${JLR} \
    ${MARGFLAGS} --consistency-lambda ${LAMBDA} \
    --batch-per-gpu ${BPG} --ms-steps 3 \
    --outdir "${OUTDIR}" --compile --ckpt-every 2
PY=$?
echo "python exit: ${PY} - $(date)"
[ "${PY}" -ne 0 ] && exit ${PY}
touch "${OUTDIR}/done.flag"

# --- inline gap test: dense GLD trim vs sparse reconciliation build ---
echo "=== GAP TEST (lambda=${LAMBDA}) -- dense trim vs sparse build ==="
gap_one () {  # $1 table  $2 node  $3 label
  python3.11 scripts/analyze_ancestral_node.py --table "$1" --node "$2" \
      --models "${OUTDIR}/model_ho3.pth" --actual-mode raw --device auto \
      --csv-out "/tmp/cons_${ARM}_${3}_l${LAMBDA}.tsv" --output /dev/null \
    && python3.11 -c "import pandas as pd; d=pd.read_csv('/tmp/cons_${ARM}_${3}_l${LAMBDA}.tsv',sep='\t'); print('  ${3} present=%d' % int((d.mean_actual>=0.5).sum()))"
}
if [ "${ARM}" = "mix" ]; then
  gap_one LACA_GLD_min1_input.tsv   LACA_GLD_min1  dense
  gap_one data/LACA_combined_table.tsv LACA        sparse
else
  gap_one LBCA_davin_sl_min1_input.tsv LBCA_davin_sl_min1 dense
  gap_one data/TableAncestralRoot1.tsv 2012              sparse
fi
echo "baseline gap (marginal): mix dense 1769 / sparse 1013 ;  bac dense ~1753 / sparse ~1524"
echo "done $(date)"
