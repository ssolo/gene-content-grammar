#!/bin/bash
# Dynamic interaction R(T) sweep for the four pilot pathogens, scored against
# both the experimental and the combined STRING targets. The plain pairwise model
# runs on all four; the higher-order (attention) model runs on the two best
# covered Proteobacteria, to test whether beyond-pairwise structure helps.
# Requires a GPU.
set -uo pipefail
cd "$(dirname "$0")/../.."
PLAIN=gsd_results_nohidden_finetune_chain_T8to20_split1/model_T16to20_f1.pth
HO=gsd_results_higher_order_nohidden_T20_split1/model_ho3.pth

for tx in 511145:E.coli 99287:S.enterica 208964:P.aeruginosa 93061:S.aureus; do
  t=${tx%%:*}; n=${tx##*:}
  echo "##### PLAIN $t $n #####"
  .venv/bin/python scripts/interactome/dynamic_interaction.py --taxon "$t" --name "$n" --ckpt "$PLAIN" || echo "PLAIN_FAIL $t"
done

for tx in 511145:E.coli 99287:S.enterica; do
  t=${tx%%:*}; n=${tx##*:}
  echo "##### HO $t $n #####"
  .venv/bin/python scripts/interactome/dynamic_interaction.py --taxon "$t" --name "$n" --ckpt "$HO" --chunk 8 --tag _ho || echo "HO_FAIL $t"
done
echo DYN_ALL_DONE
