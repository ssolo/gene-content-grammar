#!/bin/bash
#SBATCH --job-name=gate-warmstart
#SBATCH --partition=
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=gate-warmstart-%j.out
#
# Validate the WARM-START model_s3e against the matched-strength FP gate, the same
# decisive extant test used for the marginal-HQ baseline.  Submit with a dependency
# on the warm-start so it auto-runs when training finishes:
#   sbatch --dependency=afterany:<warmstart_jobid> slurm/run_gate_warmstart.sbatch
#
# Compare the printed AUCs to the marginal-HQ E. coli baseline:
#   cavity(-q_mask_absent)=0.557  self_anchor_absent=0.631  -pl_field=0.482  -dropout=0.573
# Success = self_anchor / cavity AUC clearly past ~0.65 WITH recall held (val50 R~0.8
# during training).  The warm-start retrained only bac split5 (E. coli's held-out
# split), so this validates E. coli.
set -uo pipefail
cd $PROJECT_ROOT
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi
PY=python3.11
WS='gsd_results_warmstart_fpctx_bac_split{split}/model_s3e.pth'

echo "===== WARM-START model_s3e gate -- E. coli, matched strength fn=0.9 fp=0.05 marginal ====="
$PY scripts/fp_context_diagnostics.py extant \
  --domain d__Bacteria --genome "Escherichia coli" \
  --fn 0.9 --fp 0.05 --fp-mode marginal \
  --models-template "$WS" --splits 5 \
  --dropout-runs 32 --drop-frac 0.5 --device cuda \
  --out data/gate_warmstart_ecoli.tsv

echo "===== SUMMARIZE (compare to marginal-HQ baseline cavity 0.557 / self_anchor 0.631) ====="
$PY scripts/fp_context_diagnostics.py summarize data/gate_warmstart_ecoli.tsv
echo "gate-warmstart done"
