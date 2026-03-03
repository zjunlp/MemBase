#!/usr/bin/env bash
# Stage 3: Question Answering & Evaluation for A-MEM on LongMemEval.
# Please modify the variables below to fit your setup.
# ========================================================
search_results_path="amem_output/10_0_4.json"
qa_model="gpt-4.1-mini"
judge_model="gpt-4.1-mini"
qa_batch_size=4
judge_batch_size=4
dataset_type="LongMemEval"
api_config_path="examples/evaluate_amem_on_longmemeval/api_config.json"
# ========================================================

python memory_evaluation.py \
    --search-results-path "$search_results_path" \
    --qa-model "$qa_model" \
    --judge-model "$judge_model" \
    --qa-batch-size "$qa_batch_size" \
    --judge-batch-size "$judge_batch_size" \
    --dataset-type "$dataset_type" \
    --api-config-path "$api_config_path"
