#!/bin/bash
set -e

mkdir -p logs

GRAPH_HIDDENS=(32 64 128 256)
MAX_LENS=(64 128 256)
SEEDS=(0,1,2)

EPOCHS=30

for ml in "${MAX_LENS[@]}"; do
  for gh in "${GRAPH_HIDDENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running max_len=${ml}, gh=${gh}, seed=${seed}"
      python train_multiclass_engagement_e2e_hparam.py \
        --max_len ${ml} \
        --graph_hidden ${gh} \
        --epochs ${EPOCHS} \
        --seed ${seed} \
        > logs/maxlen_${ml}_gh_${gh}_seed_${seed}.log 2>&1
    done
  done
done
