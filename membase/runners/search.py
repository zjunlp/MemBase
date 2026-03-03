import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pydantic import (
    BaseModel, 
    ConfigDict, 
    Field,
)
from tqdm import tqdm
from ..configs import CONFIG_MAPPING
from ..datasets import DATASET_MAPPING
from ..layers import MEMORY_LAYERS_MAPPING
from ..model_types.dataset import QuestionAnswerPair
from ..model_types.memory import MemoryEntry
from typing import Any, Callable


_LOCK = threading.Lock()


def memory_search(
    layer_type: str,
    user_id: str,
    questions: list[QuestionAnswerPair],
    config: dict[str, Any] | None = None,
    top_k: int = 10,
    strict: bool = True, 
) -> list[dict[str, Any]]:
    """Search memories for a given user based on questions.

    This function is designed to be submitted to a thread pool so that
    multiple users can be processed in parallel.

    Args:
        layer_type (`str`): 
            The registered name of the memory layer.
        user_id (`str`): 
            Unique identifier of the user whose memory is being searched.
        questions (`list[QuestionAnswerPair]`): 
            The evaluation questions for this user.
        config (`dict[str, Any] | None`, optional): 
            Raw configuration dictionary for the memory layer.
        top_k (`int`, defaults to `10`): 
            Number of memories to retrieve for each query.
        strict (`bool`, defaults to `True`): 
            If `True`, raise an error when no memory is found for the user.
            If `False`, return a placeholder entry instead.

    Returns:
        `list[dict[str, Any]]`: 
            A list of retrieval results, each containing the retrieved memories, 
            the question-answer pair, and the user id.
    """
    config = deepcopy(config) or {}
    config["user_id"] = user_id
    config["save_dir"] = f"{config['save_dir']}/{user_id}" 
    
    # Load memory layer configuration and class using lazy mapping.
    config_cls = CONFIG_MAPPING[layer_type]
    config = config_cls(**config)
    
    with _LOCK:
        layer_cls = MEMORY_LAYERS_MAPPING[layer_type]
        layer = layer_cls(config)

    # Load the pre-built memory.
    with _LOCK:
        if not layer.load_memory(user_id):
            msg = f"No memory is found for user '{user_id}'."
            if strict:
                raise ValueError(msg)
            else:
                # For some baselines, there are a few cases 
                # these baselines cannot process without throwing an error.
                # We simply return an empty memory for these cases.
                print(msg)
                return [
                    {
                        "retrieved_memories": [
                            MemoryEntry(
                                content="[NO RETRIEVED MEMORIES]",
                                formatted_content="[NO RETRIEVED MEMORIES]",
                                metadata={},
                            ).model_dump(mode="python")
                        ],
                        "qa_pair": qa_pair.model_dump(mode="python"),
                        "user_id": user_id,
                    }
                    for qa_pair in questions 
                ]
    
    retrievals = []
    total_q = len(questions) 
    pbar = tqdm(
        questions,
        total=total_q,
        desc=f"🧑 User Identifier: {user_id} | 🧠 Memory Layer Type: {layer_type}",
        leave=False,      
    )

    # Perform retrieval for each question.
    for qa_pair in pbar:
        query = qa_pair.question
        # Perform retrieval using the unified interface.
        retrieved_memories = layer.retrieve(query, k=top_k)
        retrieval_result = {
            "retrieved_memories": [
                memory.model_dump(mode="python")
                for memory in retrieved_memories
            ],
            "qa_pair": qa_pair.model_dump(mode="python"),
            "user_id": user_id,
        }
        retrievals.append(retrieval_result)
    
    with _LOCK:
        layer.cleanup()
    
    return retrievals


class SearchRunnerConfig(BaseModel):
    """Configuration for the search runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_type: str = Field(
        ...,
        description="The type of the memory layer to be searched.",
    )
    dataset_type: str = Field(
        ...,
        description="The type of the dataset used to search the memory layer.",
    )
    dataset_path: str = Field(
        ...,
        description="The path to the dataset.",
    )
    dataset_standardized: bool = Field(
        default=False,
        description="Whether the dataset is already standardized.",
    )
    config_path: str | None = Field(
        default=None,
        description="Path to JSON config for memory method.",
    )
    memory_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Dictionary config for the memory method. "
            "If provided, it takes precedence over ``config_path``."
        ),
    )
    num_workers: int = Field(
        default=4,
        description="The number of threads to use for the search.",
    )
    top_k: int = Field(
        default=10,
        description="Number of memories to retrieve for each query.",
    )
    start_idx: int | None = Field(
        default=None,
        description="The starting index of the trajectories to be processed.",
    )
    end_idx: int | None = Field(
        default=None,
        description="The ending index of the trajectories to be processed.",
    )
    strict: bool = Field(
        default=False,
        description="Whether to raise an error if no memory is found for a user.",
    )
    question_filter: Callable[[QuestionAnswerPair], bool] | None = Field(
        default=None,
        description="A callable that filters the evaluation questions.",
    )


class SearchRunner:
    """Runner that orchestrates the memory retrieval stage.

    It loads the dataset, dispatches per-user retrieval tasks to a thread pool,
    and saves the aggregated results to a JSON file.
    """

    def __init__(self, config: SearchRunnerConfig) -> None:
        """Initialize the search runner.

        Args:
            config (`SearchRunnerConfig`): 
                The runner configuration.
        """
        self.config = config

    def _resolve_memory_config(self) -> dict[str, Any] | None:
        """Return the memory configuration dictionary."""
        if self.config.memory_config is not None:
            return self.config.memory_config
        if self.config.config_path is not None:
            with open(self.config.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def run(self) -> list[dict[str, Any]]:
        """Execute the memory retrieval pipeline.

        Returns:
            `list[dict[str, Any]]`: 
                A list of retrieval results. Each element is a dictionary 
                containing the retrieved memories, the question-answer pair, and the user id.
        """
        cfg = self.config
        config = self._resolve_memory_config()

        # Prepare the dataset using lazy mapping.
        ds_cls = DATASET_MAPPING[cfg.dataset_type]
        if cfg.dataset_standardized:
            dataset = ds_cls.read_dataset(cfg.dataset_path)
        else:
            dataset = ds_cls.read_raw_data(cfg.dataset_path)
        if cfg.question_filter is not None:
            dataset = dataset.sample(question_filter=cfg.question_filter)
            dataset.save_dataset(f"{config['save_dir']}/{cfg.dataset_type}_stage_2")
        print("✅ The dataset is loaded successfully.")

        # Resolve index range.
        start_idx = cfg.start_idx if cfg.start_idx is not None else 0
        end_idx = cfg.end_idx if cfg.end_idx is not None else len(dataset)
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(dataset))
        if start_idx >= end_idx:
            raise ValueError("The starting index must be less than the ending index.")

        # Dispatch per-user retrieval to a thread pool.
        print("🔍 Searching memories for each trajectory...")
        retrievals = []
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            futures = []
            for trajectory, qa_pairs in zip(*dataset[start_idx:end_idx]):
                user_id = trajectory.id
                future = executor.submit(
                    memory_search,
                    cfg.memory_type,
                    user_id,
                    qa_pairs,
                    config=config,
                    top_k=cfg.top_k,
                    strict=cfg.strict,
                )
                futures.append(future)
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="🔍 Searching memories",
            ):
                results = future.result()
                retrievals.extend(results)

        # Save the retrieval results.
        output_path = (
            f"{config['save_dir']}/{cfg.top_k}_{start_idx}_{end_idx}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                retrievals,
                f,
                ensure_ascii=False,
                indent=4,
            )
        print(f"✅ {len(retrievals)} retrieval results are saved to {output_path}.")

        return retrievals
