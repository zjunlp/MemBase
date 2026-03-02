import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading
from copy import deepcopy
from membase import (
    CONFIG_MAPPING,
    MEMORY_LAYERS_MAPPING,
    DATASET_MAPPING,
)
from membase.model_types.dataset import QuestionAnswerPair
from membase.model_types.memory import MemoryEntry
from membase.utils.files import import_function_from_path
from typing import Any


_LOCK = threading.Lock()


def memory_search(
    layer_type: str,
    user_id: str,
    questions: list[QuestionAnswerPair],
    config: dict[str, Any] | None = None,
    top_k: int = 10,
    strict: bool = True, 
) -> list[dict[str, Any]]:
    """Search memories for a given user based on questions."""
    config = config or {}
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to search memories for a given user based on questions."
    )
    parser.add_argument(
        "--memory-type",
        choices=list(MEMORY_LAYERS_MAPPING.keys()),
        type=str,
        required=True,
        help="The type of the memory layer to be searched."
    )
    parser.add_argument(
        "--dataset-type",
        choices=list(DATASET_MAPPING.keys()),
        type=str,
        required=True,
        help="The type of the dataset used to search the memory layer."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="The path to the dataset."
    )
    parser.add_argument(
        "--dataset-standardized",
        action="store_true",
        help="Whether the dataset is already standardized."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="The number of threads to use for the search."
    )
    parser.add_argument(
        "--question-filter-path",
        type=str,
        default=None,
        help=(
            "Path to a question filter function used to filter the dataset. "
            "It supports two formats: 'module.submodule.function_name' or "
            "'path/to/file.py:function_name'."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to JSON config for memory method."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memories to retrieve for each query."
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=None,
        help="The starting index of the trajectories to be processed."
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=None,
        help="The ending index of the trajectories to be processed."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Whether to raise an error if no memory is found for a user."
    )
    args = parser.parse_args()

    # Load configuration.
    config = None
    if args.config_path is not None:
        with open(args.config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)

    # Prepare the dataset using lazy mapping.
    ds_cls = DATASET_MAPPING[args.dataset_type]
    if args.dataset_standardized:
        dataset = ds_cls.read_dataset(args.dataset_path)
    else:
        dataset = ds_cls.read_raw_data(args.dataset_path)
    if args.question_filter_path is not None:
        question_filter = import_function_from_path(args.question_filter_path)
        dataset = dataset.sample(question_filter=question_filter)
        dataset.save_dataset(f"{config['save_dir']}/{args.dataset_type}_stage_2")
    print("✅ The dataset is loaded successfully.")
    
    # Process index range.
    if args.start_idx is None:
        args.start_idx = 0
    if args.end_idx is None:
        args.end_idx = len(dataset)
    args.start_idx, args.end_idx = max(0, args.start_idx), min(args.end_idx, len(dataset))
    if args.start_idx >= args.end_idx:
        raise ValueError("The starting index must be less than the ending index.")
    
    # Perform memory search for each trajectory.
    print("🔍 Searching memories for each trajectory...")
    retrievals = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for trajectory, qa_pairs in zip(*dataset[args.start_idx: args.end_idx]):
            user_id = trajectory.id
            future = executor.submit(
                memory_search,
                args.memory_type,
                user_id,
                qa_pairs,
                config=deepcopy(config),
                top_k=args.top_k,
                strict=args.strict,
            )
            futures.append(future)
        for future in tqdm(
            as_completed(futures), 
            total=len(futures), 
            desc="🔍 Searching memories"
        ):
            results = future.result()
            retrievals.extend(results)

    output_path = f"{config['save_dir']}/{args.top_k}_{args.start_idx}_{args.end_idx}.json"

    with open(output_path, 'w', encoding="utf-8") as f:
        json.dump(
            retrievals, 
            f, 
            ensure_ascii=False, 
            indent=4,
        )
    print(f"✅ {len(retrievals)} retrieval results are saved to {output_path}.")