# Trace Memory Lifecycle

This example shows how to trace a memory system with MemBase and collect execution graphs for the full workflow: memory construction, memory retrieval, and answering user questions with retrieved memories. The configs and scripts in this example are also the reproduction code used in the [MemTrace paper](https://arxiv.org/pdf/2605.28732) to generate the execution-graph data for **MemTraceBench**.

---

## Step 1: Prepare the Environment

Different memory systems have different dependencies. Use a separate Python environment for each memory system when running this tracing workflow.

### Option A: Install with pip

```bash
conda create -n membase_mem0 python=3.12 -y
conda activate membase_mem0
pip install -r envs/mem0_requirements.txt
pip install vllm
pip install nltk
```

Replace the requirements file for other memory systems:

| Memory System | Requirements File |
|--------------|-------------------|
| Mem0 | `envs/mem0_requirements.txt` |
| EverMemOS | `envs/evermemos_requirements.txt` |
| NaiveRAG | `envs/rag_requirements.txt` |
| Long-Context | `envs/long_context_requirements.txt` |

### Option B: Install with uv

```bash
conda create -n membase_mem0 python=3.12 -y
conda activate membase_mem0
pip install uv
uv pip install -r envs/mem0_requirements.txt
uv pip install vllm
uv pip install nltk
```

---

## Step 2: Prepare Models and Data

This example uses local vLLM servers for Qwen3 embedding and reranking models. See the related [tutorial](../download_models/) for how to download it.

Start the embedding server:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve pretrained_models/Qwen3-Embedding-4B \
    --port 8008 \
    --served-model-name Qwen3-Embedding-4B \
    --gpu-memory-utilization 0.4 \
    --hf_overrides '{"is_matryoshka": true}'
```

Start the reranker server in a separate terminal when using `EverMemOS`:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve pretrained_models/Qwen3-Reranker-4B \
    --port 8001 \
    --served-model-name Qwen3-Reranker-4B \
    --gpu-memory-utilization 0.4 \
    --task score
```

Place datasets under this example directory:

| Dataset | Expected path |
|---------------|---------------|
| LoCoMo | `examples/trace_memory_lifecycle_with_membase/data/LoCoMo/locomo10.json` |
| RealMem | `examples/trace_memory_lifecycle_with_membase/data/RealMem/` |
| LongMemEval | `examples/trace_memory_lifecycle_with_membase/data/LongMemEval/longmemeval_s_cleaned.json` |

Dataset sources:

- LoCoMo: https://github.com/snap-research/locomo/tree/main/data
- RealMem: https://github.com/AvatarMemory/RealMemBench/tree/main/dataset
- LongMemEval: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

---

## Step 3: Configure the Run

Each script has two variables near the top:

```bash
METHOD="evermemos"
DATASET="locomo"
```

Supported values:

- `METHOD`: `mem0`, `evermemos`, `rag`, `long_context`
- `DATASET`: `locomo`, `realmem`, `longmemeval`

The memory system config is loaded from `examples/trace_memory_lifecycle_with_membase/configs/`.

Before running, you should edit the corresponding memory-system config and replace API keys, base URLs, model names, and save directory as needed. You should also update `api_config.json` with the API keys and base URLs used by the memory evaluation. The scripts pass `--tracing` to the unified MemBase entrypoints, so [smartcomment](https://github.com/zjunlp/smartcomment) execution graphs are exported during each stage.

---

## Step 4: Run Memory Construction

```bash
bash examples/trace_memory_lifecycle_with_membase/run_traced_construction.sh
```

The outputs include:

- The constructed memory state under the selected config's `save_dir`.
- A standardized stage-1 dataset at `<save_dir>/<DatasetType>_stage_1.json`.
- Per-user construction graphs at `<save_dir>/traced_data/<user_id>/graph_construction.json`.
- Token-cost statistics at `<save_dir>/token_cost_traced_<METHOD>.json`.

For `RealMem`, online task evaluation is also run during construction when `realmem_eval_config.json` is provided. This file is located under `examples/trace_memory_lifecycle_with_membase/configs/`.

> [!TIP] 
> You can change `sample_size` in `run_traced_construction.sh` to process more trajectories. The current value is `1`. If `sample_size` is larger than `1`, you can also increase `num_workers` to speed up construction with parallel processing. To reproduce the execution-graph data generation in the MemTrace paper, set `sample_size` to `4` for LoCoMo, `200` for LongMemEval, and `3` for RealMem.

---

## Step 5: Run Memory Retrieval

Skip this step if the selected dataset is RealMem.

```bash
bash examples/trace_memory_lifecycle_with_membase/run_traced_search.sh
```

The outputs include:

- Search results at `<save_dir>/<top_k>_<start_idx>_<end_idx>.json`.
- Per-user search graphs at `<save_dir>/traced_data/<user_id>/graph_search.json`.

The search graph imports and extends the user's memory construction graph, linking each retrieval query to the memory units returned by the memory layer.

> [!TIP] 
> You can adjust `num_workers` in `run_traced_search.sh` to speed up retrieval with parallel processing. You can also change `top_k` to control how many memory units are retrieved for each question. To reproduce the execution-graph data generation in the MemTrace paper, set `top_k` to `10`. For `RealMem`, retrieval happens during memory construction, so its `top_k` should be configured in `realmem_eval_config.json`.

---

## Step 6: Run Evaluation

Skip this step if the selected dataset is RealMem.

```bash
bash examples/trace_memory_lifecycle_with_membase/run_traced_evaluation.sh
```

The outputs include:

- Evaluation results at `<search_results_path>_evaluation.json`.
- Per-user evaluation graphs at `<save_dir>/traced_data/<user_id>/graph_evaluation.json`.

The evaluation graph imports and extends the user's search graph, tracing how retrieved memories are used in the response stage, and how the LLM-as-a-Judge metrics are computed. 

> [!TIP] 
> You can adjust `qa_model` for question answering and `judge_model` for LLM-as-a-Judge evaluation in `run_traced_evaluation.sh`. You can also tune `qa_batch_size` and `judge_batch_size` based on your API limits. Make sure `top_k` matches the value used in `run_traced_search.sh`, and set `end_idx` according to the `sample_size` used in `run_traced_construction.sh`. To reproduce the execution-graph data generation in the MemTrace paper, use GPT-4.1 mini as the question-answering model and Claude Opus 4.5 as the judge model. For `RealMem`, memory evaluation is completed during memory construction, and its question-answering model, judge model, API config path can be adjusted in `realmem_eval_config.json`.

