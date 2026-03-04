import sys
import os

sys.path.insert(
    0, 
    os.path.join(
        os.path.dirname(__file__), 
        "..", 
        ".."
    )
)

from membase.utils import download_models


REPO_IDS = [
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Reranker-4B",
]

PARENT_DIR = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "..", 
    "pretrained_models"
)


if __name__ == "__main__":
    download_models(
        repo_ids=REPO_IDS,
        parent_dir=PARENT_DIR,
    )
