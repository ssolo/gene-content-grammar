#!/bin/bash
#SBATCH --job-name=gsd-marg-a
#SBATCH --output=gsd-marg-a-%A_%a.out
#SBATCH --error=gsd-marg-a-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --requeue
#
# 1-GPU-per-split ARRAY of the marginal-FP HQ fine-tune (research; off-Zenodo).
# free an 80 GB GPU are fragmented (e.g. 1+3 across nodes), so 4-GPU jobs PEND;
# 1-GPU array tasks grab the free GPUs immediately and ramp to the 8-an 80 GB GPU assoc
# cap via --array=...%8. Resume-safe (skips done splits; trainer resumes from
# progress.json on requeue). No destructive defaults.
#
# ARM=bac init bac-FT-fp-HQ + HQ feathers -> ..._bac_fp_marginal_hq_split{N}
# ARM=mix init mix-FT-fp-HQ + HQ feathers -> ..._mix_fp_marginal_hq_split{N}
# Launch (bac, 10 splits, <=8 concurrent):
# sbatch --array=1-10%8 --export=ALL,ARM=bac slurm/run_fp_marginal_array.sh

set -uo pipefail
cd $PROJECT_ROOT

ARM=${ARM:-bac}
SPLIT=${SLURM_ARRAY_TASK_ID:?run via --array=1-10%8}
BPG=${BPG:-32} # lower for a 16 GB GPU (less memory), e.g. BPG=16

case "${ARM}" in
 mix) INIT=gsd_results_higher_order_nohidden_T20_mix_fp_real_hq_split${SPLIT}/model_ho3.pth
 TF=data/COG_mix_hq_train${SPLIT}_phylum.feather
 VF=data/COG_mix_hq_val${SPLIT}_phylum.feather
 OUTDIR=gsd_results_higher_order_nohidden_T20_mix_fp_marginal_hq_split${SPLIT} ;;
 bac) INIT=gsd_results_higher_order_nohidden_T20_bac_fp_real_hq_split${SPLIT}/model_ho3.pth
 TF=data/COG_bac_hq_train${SPLIT}_phylum.feather
 VF=data/COG_bac_hq_val${SPLIT}_phylum.feather
 OUTDIR=gsd_results_higher_order_nohidden_T20_bac_fp_marginal_hq_split${SPLIT} ;;
 *) echo "bad ARM=${ARM}"; exit 2 ;;
esac

echo "FP-MARGINAL-array ARM=${ARM} split=${SPLIT} on $(hostname) - $(date)"
echo "init=${INIT} outdir=${OUTDIR}"

if [ -f "${OUTDIR}/done.flag" ]; then echo "[skip] ${OUTDIR} already done"; exit 0; fi
if [ ! -f "${INIT}" ]; then echo "[no-init] ${INIT} missing"; exit 1; fi

if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh; module load python/3.11.11; fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# per-split inductor cache so concurrent array tasks don't race on --compile
export TORCHINDUCTOR_CACHE_DIR=$(pwd)/.torch_inductor_cache_${ARM}${SPLIT}
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${OUTDIR}"
[ -f ${OUTDIR}/progress.json ] && echo "Resuming: $(cat ${OUTDIR}/progress.json)"

python3.11 -m torch.distributed.run --standalone --nproc_per_node=gpu \
 train_higher_order_nohidden.py \
 --init-from "${INIT}" \
 --new-T 20 \
 --train-feather "${TF}" --val-feather "${VF}" \
 --module-matrix data/module_matrix_kegg.pt \
 --ho1-epochs 0 \
 --ho2-epochs 6 --ho2-lr 8e-6 \
 --ho3-epochs 4 --ho3-lr 2e-6 \
 --j-lr-frac 0.0 \
 --ho-fn-curriculum beta_hard \
 --fp-max 0.1 --fp-dist beta_low \
 --mod-inject-rate 0.1 --mod-swap-rate 0.2 --mod-swap-max 3 \
 --marginal-fp-rate 0.5 --marginal-fp-max 0.6 --marginal-fp-fn 0.15 \
 --K 4 --aux-lambda 0.0 --ms-steps 3 --batch-per-gpu ${BPG} \
 --outdir "${OUTDIR}" \
 --loss elbo --pl-alpha 0.3 --compile --ckpt-every 5
echo "Python exit: $? - $(date)"
