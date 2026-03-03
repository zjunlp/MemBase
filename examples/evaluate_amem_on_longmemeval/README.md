# Evaluate A-MEM on LongMemEval

This example walks through a complete evaluation of the **A-MEM** memory layer on the **LongMemEval** dataset using the three-stage pipeline: memory construction, memory retrieval, and question answering with evaluation.

You can replace A-MEM with any other supported memory layer by swapping the config and `--memory-type` argument.

---

## Step 1: Download the Dataset

Download the LongMemEval dataset from HuggingFace:

> https://huggingface.co/datasets/xiaowu0162/longmemeval

Save it to a local path, e.g., `/path/to/longmemeval.json`.

---

## Step 2: Prepare Configuration Files

### Memory Configuration

Each memory layer requires its own configuration JSON file. This example uses A-MEM. See [`amem_config.json`](amem_config.json).

> **Note**: The `user_id` field is a placeholder that will be overwritten during execution. API keys and base URLs are read from the environment variables `OPENAI_API_KEY` and `OPENAI_API_BASE` by default. You can also set them explicitly via `llm_api_key` / `llm_base_url` and `embedding_api_key` / `embedding_base_url` in the config if needed.

The full list of configuration fields can be found in `membase/configs/amem.py`.

### API Configuration (for Evaluation)

The evaluation stage requires an API config to call LLM-based QA and judge models. See [`api_config.json`](api_config.json):

```json
{
    "api_keys": ["sk-your-api-key-1", "sk-your-api-key-2"],
    "base_urls": ["https://api.openai.com/v1", "https://api.openai.com/v1"]
}
```

Alternatively, set environment variables instead of using `--api-config-path`:

```bash
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"
```

---

## Step 3: Run Memory Construction (Stage 1)

Edit [`run_construction.sh`](run_construction.sh) to set your dataset path, API keys, and base URLs, then run:

```bash
bash examples/evaluate_amem_on_longmemeval/run_construction.sh
```

This example processes **4 trajectories** split across **2 parallel processes** (ranges `0-2` and `2-4`), each with `num_workers=2`. Monitor progress in `amem_logs/`.

---

## Step 4: Run Memory Retrieval (Stage 2)

After memory construction completes, edit [`run_search.sh`](run_search.sh) and run:

```bash
bash examples/evaluate_amem_on_longmemeval/run_search.sh
```

The output will be saved to `{save_dir}/{top_k}_{start_idx}_{end_idx}.json` (e.g., `amem_output/10_0_4.json`).

---

## Step 5: Run Evaluation (Stage 3)

Edit [`run_evaluation.sh`](run_evaluation.sh) and run:

```bash
bash examples/evaluate_amem_on_longmemeval/run_evaluation.sh
```

The evaluation results will be saved as `{search_results_path}_evaluation.json`.

---

## Tips

1. **API Rate Limits**: Set `num_workers` conservatively (e.g., 4-8) to avoid upstream API overload.
2. **Resume Interrupted Runs**: If the process is interrupted, simply re-run the same command. Completed trajectories will be skipped automatically.
3. **Token Cost Tracking**: Check the generated `token_cost_*.json` files for detailed token consumption statistics.
4. **Log Files**: Monitor `{log_dir}/process_*.log` files for real-time progress and debugging.
