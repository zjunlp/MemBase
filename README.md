# MemBase

A comprehensive evaluation framework for benchmarking various memory layers on long-term conversational memory tasks. This framework provides a unified pipeline for **memory construction**, **memory retrieval**, and **question answering evaluation**.

---

## Key Features

- **Checkpoint, Recovery & Rerun**: It automatically saves progress during memory construction. If interrupted, simply re-run the script and it will skip already-processed trajectories and resume from where it left off. Use the `--rerun` flag to force rebuild memories from scratch when needed.
- **Non-Invasive Token Cost Monitoring**: Built-in token consumption tracking for LLM API calls. It uses monkey-patching to intercept calls **without modifying any baseline's internal code**.
- **Modular Architecture**: Clean separation between memory layers, datasets, and evaluation logic. Adding a new memory layer only requires implementing the `MemBaseLayer` interface and registering it in the `membase` package. New datasets can be added by subclassing `MemBaseDataset` and registering them.
- **Multiple Baselines & Datasets**: See [Supported Memory Layers](#supported-memory-layers) and [Supported Datasets](#supported-datasets) below.

---

## Project Structure

```
MemBase/
├── memory_construction.py       # CLI: Stage 1 – Build memories from trajectories
├── memory_search.py             # CLI: Stage 2 – Retrieve memories for each query
├── memory_evaluation.py         # CLI: Stage 3 – Answer questions and evaluate
├── envs/                        # Requirements for each baseline
├── examples/                    # Usage examples and tutorials
└── membase/                     # Core package
    ├── __init__.py              # Package-level re-exports
    ├── runners/                 # Runner classes for programmatic pipeline execution
    ├── configs/                 # Configuration classes for each memory layer
    ├── datasets/                # Dataset loaders
    ├── layers/                  # Memory layer implementations
    ├── baselines/               # Vendored baseline source code
    ├── inference_utils/         # QA and evaluation operators
    ├── model_types/             # Data models (dataset, memory)
    └── utils/                   # Token monitoring, monkey-patching, file utilities
```

---

## Installation

### Prerequisites

- **Python >= 3.12** is required.
- **Conda** (Anaconda or Miniconda) is recommended for environment management.

### Setting Up the Environment

> ⚠️ **Important**: Different memory baselines may have **conflicting dependencies**. We strongly recommend creating a **separate virtual environment for each baseline** to avoid dependency conflicts.

Each memory baseline has its own requirements file in the `envs/` directory. Below are two examples:

**Example: Setting up environment for A-MEM**

```bash
conda create -n amem_env python=3.12 -y
conda activate amem_env
pip install -r envs/amem_requirements.txt
```

**Example: Setting up environment for EverMemOS**

```bash
conda create -n evermemos_env python=3.12 -y
conda activate evermemos_env
pip install -r envs/evermemos_requirements.txt
```

Repeat the same pattern for other baselines using the corresponding requirements file in `envs/`.

---

## Evaluation Pipeline Overview

The evaluation of all memory baselines follows a **three-stage pipeline**:

### Stage 1: Memory Construction

User interaction trajectories are fed **incrementally** (message by message) into the memory layer, which builds and updates its internal memory state as each message arrives.

### Stage 2: Memory Retrieval

Given the constructed memory, this stage retrieves the top-k most relevant memory units for each evaluation query.

### Stage 3: Question Answering & Evaluation

Using the retrieved memories as context, a question-answering model generates answers for each question. A judge model then evaluates whether the generated answers match the ground truth, producing final accuracy metrics.

---

## Supported Memory Layers

- [A-MEM](https://github.com/WujiangXu/A-mem-sys)
- [LangMem](https://github.com/langchain-ai/langmem)
- [MemOS](https://github.com/MemTensor/MemOS)
- [EverMemOS](https://github.com/EverMind-AI/EverMemOS)
- [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG)
- [Long-Context](membase/layers/long_context.py)
- [NaiveRAG](membase/layers/naive_rag.py)

---

## Supported Datasets

- [LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
- [LoCoMo](https://github.com/snap-research/locomo/tree/main/data)

---

## Examples

See the [`examples/`](examples/) directory for step-by-step tutorials:

| Example | Description |
|---------|-------------|
| [Evaluate A-MEM on LongMemEval](examples/evaluate_amem_on_longmemeval/) | Run the full three-stage evaluation pipeline (construction, retrieval, QA) using A-MEM on LongMemEval |
| [Evaluate MemOS on LoCoMo](examples/evaluate_memos_on_locomo/) | Evaluate MemOS with vLLM-served embedding on LoCoMo, with adversarial question filtering |
| [Download Models](examples/download_models/) | Download pre-trained embedding and reranker models from Hugging Face |

---

## Programmatic API

Besides the CLI scripts, all three pipeline stages are available as importable Runner classes under `membase.runners`. This allows you to drive the evaluation from Python scripts or notebooks without shell commands.

---

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests. More baselines will be added.
