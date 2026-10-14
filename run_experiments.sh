#!/usr/bin/env bash
# Run every ablation variant for several seeds, then aggregate.  Usage:  DATASET=BUS bash run_experiments.sh
set -e
DATASET=${DATASET:-BUS}
SEEDS=${SEEDS:-"1 2 3 4 5"}
VARIANTS=${VARIANTS:-"full no_tvgcf no_oga no_residual no_ooa"}
for v in $VARIANTS; do
  for s in $SEEDS; do
    echo "=== $DATASET | $v | seed $s ==="
    DATASET=$DATASET VARIANT=$v SEED=$s \
      jupyter nbconvert --to notebook --execute mainCode.ipynb --output "executed_${DATASET}_${v}_seed${s}.ipynb" \
      --output-dir "${OUT_ROOT:-./results}/_executed" --ExecutePreprocessor.timeout=-1
  done
done
python aggregate_results.py --results "${OUT_ROOT:-./results}" --dataset "$DATASET"
