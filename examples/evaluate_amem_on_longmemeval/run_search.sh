#!/usr/bin/env bash
# Stage 2: Memory Retrieval for A-MEM on LongMemEval.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="A-MEM"
dataset_type="LongMemEval"
dataset_path="YOUR_DATASET_PATH"
config_path="examples/evaluate_amem_on_longmemeval/amem_config.json"
num_workers=2
top_k=10
start_idx=0
end_idx=4
# ========================================================

python memory_search.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --top-k "$top_k" \
    --start-idx "$start_idx" \
    --end-idx "$end_idx"
