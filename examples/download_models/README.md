# Download Models

This example downloads pre-trained embedding and reranker models from Hugging Face using the `download_models` utility in `membase.utils.files`.

By default, the following models are downloaded to the `pretrained_models/` directory at the project root:

| Model | Hugging Face Repo |
|-------|-------------------|
| Qwen3-Embedding-4B | [Qwen/Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| Qwen3-Reranker-4B | [Qwen/Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) |

---

## Prerequisites

Make sure `huggingface-hub` is installed:

```bash
pip install huggingface-hub
```

> **Note**: If you are downloading gated models or private repositories, log in first via `huggingface-cli login`.

---

## Usage

Run the script from the project root:

```bash
python examples/download_models/download_models.py
```

After completion, the directory structure will look like:

```
pretrained_models/
├── Qwen3-Embedding-4B/
└── Qwen3-Reranker-4B/
```

---

## Customization

You can modify `download_models.py` to download different models, specify revisions, or filter files. The `download_models` utility supports several options:

```python
from membase.utils import download_models

# Download with per-repository options.
download_models(
    repo_ids={
        "Qwen/Qwen3-Embedding-4B": {"revision": "main"},
        "Qwen/Qwen3-Reranker-4B": {"allow_patterns": ["*.safetensors", "*.json"]},
    },
    parent_dir="pretrained_models",
)
```

See `membase/utils/files.py` for the full API reference.
