# Evaluate MemOS on LoCoMo

This example walks through evaluating the **MemOS** memory layer on the **LoCoMo** dataset. It covers how to configure MemOS (including deploying an embedding model locally with vLLM and setting up Neo4j), how to handle LoCoMo's two-user conversation structure, and how to filter specific question types at the retrieval stage.

---

## Prerequisites

- **Python >= 3.12** with the MemOS environment (`pip install -r envs/memos_requirements.txt`).
- **Neo4j** running locally. The easiest way is via Docker:

```bash
docker run -p7474:7474 -p7687:7687 -d -e NEO4J_AUTH=neo4j/password neo4j:latest
```

- **vLLM** installed in the environment (`pip install vllm`).
- **Qwen3-Embedding-4B** downloaded locally. See the [example](../download_models/) for how to download it.

---

## Step 1: Serve the Embedding Model with vLLM

Start the vLLM embedding server before running the pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve pretrained_models/Qwen3-Embedding-4B \
    --port 8008 \
    --served-model-name Qwen3-Embedding-4B \
    --gpu-memory-utilization 0.5
```

This exposes an OpenAI-compatible embedding endpoint at `http://localhost:8008/v1`. The MemOS config uses the `universal_api` backend to connect to it.

> **Note**: The default MemOS embedder timeout (`MOS_EMBEDDER_TIMEOUT`) is **5 seconds**, which is often too short for vLLM-served models. The shell scripts in this example set `MOS_EMBEDDER_TIMEOUT=120` to avoid timeouts.

---

## Step 2: Download the Dataset

Download the LoCoMo dataset from GitHub:

> https://github.com/snap-research/locomo/tree/main/data

Save it to a local path, e.g., `/path/to/locomo.json`.

---

## Step 3: Configure MemOS

See [`memos_config.json`](memos_config.json). The MemOS config is composed of multiple sub-configurations (extractor LLM, dispatcher LLM, embedding, graph database, chunker, etc.). For the full list of available configuration options, see the [MemOS config source code](https://github.com/MemTensor/MemOS/tree/main/src/memos/configs) and `membase/configs/memos.py`.

### Embedding Configuration

MemOS uses `EmbedderConfigFactory` to configure the embedding backend. This example uses the `universal_api` backend to connect to the vLLM-served Qwen3-Embedding-4B. For the full schema of each backend, see the [MemOS source code](https://github.com/MemTensor/MemOS/blob/main/src/memos/configs/embedder.py).

```json
"embedding_config": {
    "backend": "universal_api",
    "config": {
        "provider": "openai",
        "api_key": "EMPTY",
        "base_url": "http://localhost:8008/v1",
        "model_name_or_path": "Qwen3-Embedding-4B",
        "embedding_dims": 2560,
        "max_tokens": null
    }
}
```

`provider` must be `"openai"` for any OpenAI-compatible API (including vLLM). `api_key` can be `"EMPTY"` for local servers. `model_name_or_path` must match the `--served-model-name` in the vLLM command.

Alternatively, you can use a local Sentence Transformer model instead of vLLM:

```json
"embedding_config": {
    "backend": "sentence_transformer",
    "config": {
        "embedding_dims": 384,
        "model_name_or_path": "/path/to/all-MiniLM-L6-v2",
        "max_tokens": 256,
        "trust_remote_code": true
    }
}
```

> **Note**: When switching embedding models, remember to update `embedding_dims` accordingly.

### LLM Configuration

In this example, we directly specify `api_key` and `api_base` in the [MemOS configuration file](memos_config.json) for both components.

### Graph Database

MemOS stores its tree-structured memory in Neo4j. In this example, `use_multi_db` is set to `false`, meaning all trajectories share a single Neo4j database with `user_name` as the partition key (automatically set from `user_id`).

After memory construction completes, all memory contents are persisted to local files and the corresponding Neo4j data is automatically cleaned up. If you need to manually clear the database (e.g., after an interrupted run), use:

```bash
docker exec -it YOUR_CONTAINER_ID cypher-shell -u neo4j -p password "MATCH (n) DETACH DELETE n"
```

---

## Step 4: Run Memory Construction (Stage 1)

Edit [`run_construction.sh`](run_construction.sh) to set your dataset path and API credentials, then run:

```bash
bash examples/evaluate_memos_on_locomo/run_construction.sh
```

This samples 2 trajectories from the full LoCoMo dataset via `--sample-size` and processes them with 2 workers in a single process. The sampled dataset is automatically saved to `memos_output/LoCoMo_stage_1.json` in standardized format for subsequent stages.

LoCoMo trajectories are conversations between two users. In this example, we maintain one unified memory per trajectory (instead of one memory per speaker), and speaker identity is encoded in each message content.

---

## Step 5: Run Memory Retrieval (Stage 2)

This stage reads the standardized dataset saved by Stage 1 (with `--dataset-standardized`).

Adversarial questions are excluded via `--question-filter-path`, which points to [`question_filter.py`](question_filter.py). You can modify this filter to target other question types (see `membase/datasets/locomo.py` for all available types).

```bash
bash examples/evaluate_memos_on_locomo/run_search.sh
```

---

## Step 6: Run Evaluation (Stage 3)

Edit [`run_evaluation.sh`](run_evaluation.sh) and run:

```bash
bash examples/evaluate_memos_on_locomo/run_evaluation.sh
```

> **Note**: Make sure to set `--dataset-type LoCoMo` so that the correct judge prompt template and response parser are used.
