#!/bin/bash
#SBATCH --job-name=gsd-ah-marg1
#SBATCH --output=gsd-ah-marg1-%A_%a.out
#SBATCH --error=gsd-ah-marg1-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=6
#SBATCH --requeue
#
# SINGLE-GPU (no DDP) add-hidden marginal -- 8x1: one split per array task.
# Plain python (NO torchrun) -> single-process -> no DDP wrap (sidesteps the
# DDP wrap-order bug and is more GPU-efficient than 2x4 DDP for this comm-bound
# model).  Per-stage resume via progress.json (the script skips completed AH
# stages on rerun); a split that hits the 12h wall is re-runnable by re-sbatch.
#
#   sbatch --export=ALL,ARM=mix,AH1=8,AH2=4,AH3=4 --array=1-3%8 slurm/run_add_hidden_marginal_1gpu.sh
#   full set: --array=1-10%8 ;  bac arm: ARM=bac
# After:
#   python scripts/validate_marginal.py --model mix-FT-fp-marginal-ah-hq --splits 1-3 \
#          --gld laca --ckpt-name model_ah3.pth
set -uo pipefail
cd $PROJECT_ROOT
ARM=${ARM:?set ARM=mix|bac}
SPLIT=${SLURM_ARRAY_TASK_ID:?launch with --array=1-N}
H=${H:-100}; K=${K:-6}; BPG=${BPG:-32}
AH1=${AH1:-12}; AH2=${AH2:-15}; AH3=${AH3:-8}
ONS=${ONS:-full}   # full: U xavier-init feeds z -> A grows (transparent A=0 works). tied (U=A^T) deadlocks at A=0.
A_INIT=${A_INIT:-0.0}   # >0: init A with N(0,A_INIT) so the hidden engage from step 0 (breaks A=0 deadlock)
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

echo "1GPU ADD-HIDDEN ARM=${ARM} split=${SPLIT} H=${H} K=${K} AH=${AH1}/${AH2}/${AH3} onsager=${ONS} a_init=${A_INIT}  $(hostname) $(date)"
echo "init=${INIT} -> outdir=${OUTDIR}  marg=${MARGFLAGS} ${MARGFREQ:-(broad)}"

if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; module load python/3.11.11; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache_s${SPLIT}
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"

[ -f "${OUTDIR}/done.flag" ] && { echo "[skip] ${OUTDIR} already done"; exit 0; }
[ -s "${INIT}" ] || { echo "[no-init] ${INIT} not found"; exit 1; }
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

# PLAIN python => single GPU, single process => dist not initialised => no DDP.
python3.11 train_add_hidden.py \
    --init-from "${INIT}" --new-T 20 \
    --train-feather "${TF}" --val-feather "${VF}" \
    --module-matrix data/module_matrix_kegg.pt \
    --H ${H} --K ${K} --onsager ${ONS} --a-init-noise ${A_INIT} --loss elbo --pl-alpha 0.3 \
    --ah1-epochs ${AH1} --ah2-epochs ${AH2} --ah3-epochs ${AH3} --j-lr-frac 0.0 \
    ${MARGFLAGS} ${MARGFREQ} --batch-per-gpu ${BPG} --ms-steps 3 \
    --outdir "${OUTDIR}" --compile --ckpt-every 5
echo "python exit: $? - $(date)"
