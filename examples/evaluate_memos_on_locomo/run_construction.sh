#!/usr/bin/env bash
# Stage 1: Memory Construction for MemOS on LoCoMo.
# It samples 2 trajectories from the full dataset and processes them
# with a single process and 2 workers.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="MemOS"
dataset_type="LoCoMo"
dataset_path="YOUR_DATASET_PATH"
config_path="examples/evaluate_memos_on_locomo/memos_config.json"
num_workers=2
sample_size=2
tokenizer_path="gpt-4.1-mini"
log_dir="memos_logs"
token_cost_file="token_cost_memos"
# ========================================================

# The default timeout for MemOS embedder is 5 seconds.
# Increase it when using vLLM-served embedding models.
export MOS_EMBEDDER_TIMEOUT=120

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

log_file="${log_dir}/process_1.log"
[ ! -f "$log_file" ] && touch "$log_file"

nohup python memory_construction.py \
    --memory-type "$memory_type" \
    --dataset-type "$dataset_type" \
    --dataset-path "$dataset_path" \
    --config-path "$config_path" \
    --num-workers "$num_workers" \
    --sample-size "$sample_size" \
    --token-cost-save-filename "$token_cost_file" \
    --tokenizer-path "$tokenizer_path" \
    > "$log_file" 2>&1 &
echo $! > "${log_dir}/process_1.pid"
