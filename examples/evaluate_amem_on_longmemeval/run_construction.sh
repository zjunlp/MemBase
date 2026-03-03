#!/usr/bin/env bash
# Stage 1: Memory Construction for A-MEM on LongMemEval.
# It launches 2 parallel processes, each handling 2 trajectories.
# Please modify the variables below to fit your setup.
# ========================================================
memory_type="A-MEM"
dataset_type="LongMemEval"
dataset_path="YOUR_DATASET_PATH"
config_path="examples/evaluate_amem_on_longmemeval/amem_config.json"
num_workers=2
tokenizer_path="gpt-4.1-mini"
log_dir="amem_logs"
token_cost_prefix="token_cost"
pid_prefix="process"

ranges=(
    "0 2"
    "2 4"
)

api_keys=(
    "YOUR_API_KEY_1"
    "YOUR_API_KEY_2"
)

base_urls=(
    "YOUR_BASE_URL_1"
    "YOUR_BASE_URL_2"
)
# ========================================================

[ ! -d "$log_dir" ] && mkdir -p "$log_dir"

for ((i=0; i<${#ranges[@]}; i++)); do
    read start_idx end_idx <<< "${ranges[$i]}"
    export OPENAI_API_KEY="${api_keys[$i]}"
    export OPENAI_API_BASE="${base_urls[$i]}"

    log_file="${log_dir}/${pid_prefix}_$((i+1))_${start_idx}_${end_idx}.log"
    token_cost_file="${token_cost_prefix}_${memory_type,,}_$((i+1))_${start_idx}_${end_idx}"
    pid_file="${log_dir}/${pid_prefix}_$((i+1)).pid"

    [ ! -f "$log_file" ] && touch "$log_file"

    nohup python memory_construction.py \
        --memory-type "$memory_type" \
        --dataset-type "$dataset_type" \
        --dataset-path "$dataset_path" \
        --config-path "$config_path" \
        --num-workers "$num_workers" \
        --start-idx "$start_idx" \
        --end-idx "$end_idx" \
        --token-cost-save-filename "$token_cost_file" \
        --tokenizer-path "$tokenizer_path" \
        > "$log_file" 2>&1 &
    echo $! > "$pid_file"
    sleep 10
done
