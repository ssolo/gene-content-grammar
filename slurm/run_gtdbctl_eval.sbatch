#!/bin/bash
#SBATCH --job-name=gtdbctl-eval
#SBATCH --output=gtdbctl-eval-%A_%a.out
#SBATCH --error=gtdbctl-eval-%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#
# Coherent-FP rejection frontier for the GTDB-controls FINE-TUNES (genome-mass
# norm baked into training, analysis/gtdb_gene_content_epistasis.tex Model C).
# Tests whether the +0.055 FP-removed-at-matched-recall gain verified on
# mix-split1 REPLICATES across mix splits {1,2,3,4} and the bacterial domain
# (bac-split1).
#
# PER-SPLIT comparison (NEVER vs a pooled ensemble baseline -- see the
# analysis-provenance / consistency lessons): for each (arm, split) we evaluate
# BOTH on the SAME val set with the SAME contamination seed/grafting:
# * baseline = the consistency model that the fine-tune was init-ed FROM
# (no coupling controls; gmass_kbar=None -> norm OFF)
# * gtdbctl = the fine-tune, with --genome-mass-norm re-applied at eval.
# gmass_kbar is a PLAIN ATTRIBUTE (models.py:944), NOT a saved buffer, so the
# trained-in norm must be re-set at load; set_coupling_controls reads the
# SAME data/cog_marginal_frequency.tsv the trainer used -> identical
# kbar=sum_c p_c, so the eval forward is byte-identical to the trained model.
# The frontier gain is FP-removed(gtdbctl) - FP-removed(baseline) at matched
# recall, computed per split, then summarised across splits.
#
# a 16 GB GPU gpu partition: ADDITIVE to the 80 GB GPU cap, so this does NOT compete
# with the running strong-reg sweep (gtdb-ft on ) for GPUs.
#
# sbatch slurm/run_gtdbctl_eval.sbatch
# compare: scripts/coherent_fp_eval writes data/cohfp_gtdbctl_<arm>_<role>_s<split>.tsv
set -uo pipefail
cd $PROJECT_ROOT

# task -> (arm, split): mix{1,2,3,4} + bac{1}
ARMS=(mix mix mix mix bac)
SPLS=(1 2 3 4 1)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
S=${SPLS[$SLURM_ARRAY_TASK_ID]}

case "${ARM}" in
 mix) VG='data/COG_val{split}_phylum.feather' ;;
 bac) VG='data/COG_bac_val{split}_phylum.feather' ;;
 *) echo "bad ARM=${ARM}"; exit 2 ;;
esac
FT=gsd_results_gtdbctl_T20_${ARM}_split${S}/model_ho3.pth
BASE=gsd_results_consistency_T20_${ARM}_fp_marginal_cons_l1.0_j1.0_hq_split${S}/model_ho3.pth
OUT_FT=data/cohfp_gtdbctl_${ARM}_gtdbctl_s${S}.tsv
OUT_BASE=data/cohfp_gtdbctl_${ARM}_base_s${S}.tsv
EVAL="--val-glob ${VG} --splits ${S} --fn 0.5 0.9 --n-graft 3 --limit 4000 --seed 7 --batch 64 --device cuda"

echo "GTDBCTL-EVAL task=${SLURM_ARRAY_TASK_ID} arm=${ARM} split=${S} $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1
[ -s "${FT}" ] || { echo "[no-ft] ${FT} not found"; exit 1; }
[ -s "${BASE}" ] || { echo "[no-base] ${BASE} not found"; exit 1; }
if [ -f .venv/bin/activate ]; then source .venv/bin/activate
else module load python/3.11.11 2>/dev/null || true; fi

# baseline (controls OFF)
if [ -s "${OUT_BASE}" ]; then echo "[skip] ${OUT_BASE}"; else
 python3.11 scripts/coherent_fp_eval.py \
 --models "gsd_results_consistency_T20_${ARM}_fp_marginal_cons_l1.0_j1.0_hq_split{split}/model_ho3.pth" \
 --label "${ARM}-base-s${S}" ${EVAL} --out "${OUT_BASE}" && echo "BASE_DONE ${ARM} s${S}"
fi
# fine-tune with the trained-in genome-mass norm re-applied
if [ -s "${OUT_FT}" ]; then echo "[skip] ${OUT_FT}"; else
 python3.11 scripts/coherent_fp_eval.py \
 --models "gsd_results_gtdbctl_T20_${ARM}_split{split}/model_ho3.pth" \
 --label "${ARM}-gtdbctl-s${S}" ${EVAL} --genome-mass-norm \
 --freq-tsv data/cog_marginal_frequency.tsv --out "${OUT_FT}" && echo "FT_DONE ${ARM} s${S}"
fi
echo "done arm=${ARM} split=${S} $(date)"
