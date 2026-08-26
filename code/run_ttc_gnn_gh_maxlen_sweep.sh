#!/bin/bash
set -e

mkdir -p logs

GRAPH_HIDDENS=(32 64 128 256)
MAX_LENS=(64 128 256)
SEEDS=(0 1 2 3 4)

for gh in "${GRAPH_HIDDENS[@]}"; do
  for ml in "${MAX_LENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running gh=${gh}, max_len=${ml}, seed=${seed}"
      python train_multiclass_engagement_e2e_hparam.py \
        --graph_hidden ${gh} \
        --max_len ${ml} \
        --epochs 10 \
        --seed ${seed} \
        > logs/maxlen_${ml}_gh_${gh}_seed_${seed}.log 2>&1
    done
  done
done
