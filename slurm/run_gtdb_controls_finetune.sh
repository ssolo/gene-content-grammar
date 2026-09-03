#!/bin/bash
#SBATCH --job-name=gtdb-ft
#SBATCH --output=gtdb-ft-%A_%a.out
#SBATCH --error=gtdb-ft-%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#
# Fine-tune the cross-input-consistent marginal with the GTDB-epistasis coupling
# discipline (analysis/gtdb_gene_content_epistasis.tex, Model C) BAKED INTO
# TRAINING -- testing whether retraining gives what post-hoc could not:
#   --genome-mass-norm : sqrt(kbar/k_n) coupling-drive normalisation
#   --lowrank-lambda   : nuclear-norm penalty -> transferable low rank
#   --zeromean-lambda  : zero-mean coupling (field carries base rates)
#   --per-state-cal    : trainable gain/loss output calibration head
# Single GPU (plain python => no DDP, so the J/HO autograd hazard is avoided),
# init from the consistency model, ho3 only with J FREE (j-lr-frac 1.0).
#
#   sbatch --export=ALL,ARM=mix --array=1 slurm/run_gtdb_controls_finetune.sh
#   sweep: LOWRANK=1e-4 ZEROMEAN=3e-3 HO3=15 ; bac arm: ARM=bac
set -uo pipefail
cd $PROJECT_ROOT
ARM=${ARM:?set ARM=mix|bac}
SPLIT=${SLURM_ARRAY_TASK_ID:?launch with --array=1-N}
K=${K:-6}; BPG=${BPG:-32}; LAMBDA=${LAMBDA:-1.0}
HO3=${HO3:-12}; HO3LR=${HO3LR:-2e-6}; JLR=${JLR:-1.0}
LOWRANK=${LOWRANK:-3e-4}; ZEROMEAN=${ZEROMEAN:-1e-3}; LRANK=${LRANK:-16}
TAG=${TAG:-}   # config tag appended to OUTDIR so sweep configs do not collide
MARGFLAGS="--marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15"

case "${ARM}" in
  mix) INIT=gsd_results_consistency_T20_mix_fp_marginal_cons_l1.0_j1.0_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
       VF=data/COG_mix_hq_val${SPLIT}_phylum.feather ;;
  bac) INIT=gsd_results_consistency_T20_bac_fp_marginal_cons_l1.0_j1.0_hq_split${SPLIT}/model_ho3.pth
       TF=data/COG_bac_hq_train${SPLIT}_phylum.feather
       VF=data/COG_bac_hq_val${SPLIT}_phylum.feather ;;
  *)   echo "bad ARM=${ARM}"; exit 2 ;;
esac
OUTDIR=gsd_results_gtdbctl_T20_${ARM}_split${SPLIT}${TAG}

echo "GTDB-CTL ft ARM=${ARM} split=${SPLIT} ho3=${HO3}@${HO3LR} jlr=${JLR} lowrank=${LOWRANK} zeromean=${ZEROMEAN}  $(hostname) $(date)"
echo "init=${INIT} -> ${OUTDIR}"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; module load python/3.11.11; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${OUTDIR}"
[ -f "${OUTDIR}/done.flag" ] && { echo "[skip] done"; exit 0; }
[ -s "${INIT}" ] || { echo "[no-init] ${INIT} not found"; exit 1; }

# No --compile: the nuclear-norm SVD regulariser is eager-only.
python3.11 train_higher_order_nohidden.py \
    --init-from "${INIT}" --new-T 20 \
    --train-feather "${TF}" --val-feather "${VF}" \
    --module-matrix data/module_matrix_kegg.pt \
    --K ${K} --loss elbo --pl-alpha 0.3 \
    --ho1-epochs 0 --ho2-epochs 0 --ho3-epochs ${HO3} --ho3-lr ${HO3LR} --j-lr-frac ${JLR} \
    ${MARGFLAGS} --consistency-lambda ${LAMBDA} \
    --genome-mass-norm --per-state-cal \
    --lowrank-lambda ${LOWRANK} --zeromean-lambda ${ZEROMEAN} --lowrank-rank ${LRANK} \
    --batch-per-gpu ${BPG} --ms-steps 3 \
    --outdir "${OUTDIR}" --ckpt-every 2
PY=$?
echo "python exit: ${PY} - $(date)"
[ "${PY}" -ne 0 ] && exit ${PY}
touch "${OUTDIR}/done.flag"
echo "done $(date) -- evaluate with coherent_fp_eval.py + analyze_ancestral_node.py"
