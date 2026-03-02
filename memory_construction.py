import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
import time 
import threading
from copy import deepcopy 
import os
from membase.utils import (
    CostStateManager, 
    CostState, 
    get_tokenizer_for_model,
    MonkeyPatcher, 
    import_function_from_path,
)
from membase import (
    CONFIG_MAPPING,
    MEMORY_LAYERS_MAPPING,
    DATASET_MAPPING,
)
from membase.model_types.dataset import Message, Trajectory
from typing import Any, Callable


_LOCK = threading.Lock()


def memory_construction(
    layer_type: str, 
    user_id: str, 
    trajectory: Trajectory, 
    config: dict[str, Any] | None = None, 
    rerun: bool = False,
    message_preprocessor: Callable[[Message], Message] | None = None,
) -> dict[str, float]: 
    """Given a specific interaction trajectory, this function builds a memory."""
    config = config or {}
    # It overrides the user id in the config. 
    config["user_id"] = user_id 
    # Each user has a distinct config directory. 
    config["save_dir"] = f"{config['save_dir']}/{user_id}" 

    # Use lazy mapping to load config and layer classes.
    config_cls = CONFIG_MAPPING[layer_type]
    config = config_cls(**config)
    layer_cls = MEMORY_LAYERS_MAPPING[layer_type] 
    
    with _LOCK:
        layer = layer_cls(config)

    if message_preprocessor is None:
        message_preprocessor = lambda message: message

    output = {
        "total_add_time": 0.0,
        "avg_add_time": 0.0,
    }

    with _LOCK:
        # It includes I/O operations. 
        if not rerun and layer.load_memory(user_id):
            print(f"🔄 The memory for user '{user_id}' is loaded successfully 😄.")
            return output
    
    specs = layer.get_patch_specs() 
    with MonkeyPatcher(specs):
        total_msgs = sum(len(session) for session in trajectory) 
        pbar_desc = f"🧑 User Identifier: {user_id} | 🧠 Memory Layer Type: {layer_type}" 
        pbar = tqdm(
            total=total_msgs,
            desc=pbar_desc,
            leave=False,   
        )

        # Start to construct the memory for a specific trajectory.  
        for session in trajectory:
            for message in session:
                start_time = datetime.now() 
                message = message.model_copy(deep=True)
                message = message_preprocessor(message)
                layer.add_message(message)

                end_time = datetime.now() 
                output["total_add_time"] += (end_time - start_time).total_seconds()

                pbar.update(1) 
                time.sleep(0.2)
        pbar.close()
    
    start_time = datetime.now()   
    # It may include I/O operations (loading a sentence embedding model).
    with _LOCK:
        layer.flush()
    end_time = datetime.now()  
    output["total_add_time"] += (end_time - start_time).total_seconds() 

    # It includes I/O operations. 
    with _LOCK:
        layer.save_memory() 
        layer.cleanup()
    output["avg_add_time"] = output["total_add_time"] / len(trajectory)

    return output 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script used to evaluate various memory layers on various datasets."
    )
    parser.add_argument(
        "--memory-type", 
        choices=list(MEMORY_LAYERS_MAPPING.keys()), 
        type=str, 
        required=True, 
        help="The type of the memory layer to be evaluated."
    )
    parser.add_argument(
        "--dataset-type", 
        choices=list(DATASET_MAPPING.keys()), 
        type=str, 
        required=True, 
        help="The type of the dataset used to evaluate the memory layer."
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
        help="The number of threads to use for the evaluation."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed used to sample the dataset if the user provides the sample size."
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=None, 
        help="Subset size from dataset."
    )
    parser.add_argument(
        "--rerun", 
        action="store_true", 
        help="Ignore saved memory and rebuild the memory from scratch."
    )
    parser.add_argument(
        "--config-path", 
        type=str, 
        default=None,
        help="Path to the JSON config for the memory method."
    )
    parser.add_argument(
        "--token-cost-save-filename", 
        type=str, 
        default="token_cost", 
        help="Path to save the statistics related to the token consumption."
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
        "--tokenizer-path", 
        type=str, 
        default=None, 
        help="The path to the tokenizer (only for backbone model)."
    )
    parser.add_argument(
        "--message-preprocessor-path", 
        type=str, 
        default=None, 
        help=(
            "Path to a callable that preprocesses each message before it is added to the memory. "
            "It supports two formats: (1) Python module path, e.g. 'mypackage.module.func'; "
            "(2) file path with function name, e.g. 'path/to/file.py:func'."
        ),
    )
    parser.add_argument(
        "--sample-filter-path", 
        type=str, 
        default=None, 
        help=(
            "Path to a callable that filters dataset samples. "
            "It supports two formats: (1) Python module path, e.g. 'mypackage.module.func'; "
            "(2) file path with function name, e.g. 'path/to/file.py:func'."
        ),
    )
    args = parser.parse_args()

    message_preprocessor = None
    if args.message_preprocessor_path is not None:
        message_preprocessor = import_function_from_path(args.message_preprocessor_path)
        print(f"A message preprocessor is loaded from '{args.message_preprocessor_path}'.")

    sample_filter = None
    if args.sample_filter_path is not None:
        sample_filter = import_function_from_path(args.sample_filter_path)
        print(f"A sample filter is loaded from '{args.sample_filter_path}'.")

    config = None 
    if args.config_path is not None:
        with open(args.config_path, 'r', encoding="utf-8") as f:
            config = json.load(f)

    # Get a dummy configuration to infer the corresponding LLM being used.
    # Use lazy mapping to load config class.
    config_cls = CONFIG_MAPPING[args.memory_type]
    if config is None:
        dummy_user_id = "guest" 
        dummy_config = config_cls(user_id=dummy_user_id)
    else:
        dummy_config = config_cls(**config) 
    
    # If token cost file exists, we load it. 
    if os.path.exists(args.token_cost_save_filename + ".json"):
        with open(args.token_cost_save_filename + ".json", 'r') as f:
            token_cost = json.load(f)
        for model, state in token_cost.items():
            is_dict = all(isinstance(value, dict) for value in state.values())
            if not is_dict:
                token_cost[model] = CostState.from_dict(state)
            else:
                token_cost[model] = {
                    op: CostState.from_dict(cs) for op, cs in state.items()
                }
    else: 
        token_cost = {} 
    
    # Before run the expriment, we should register the base model being used. 
    # Please ensure all types of config classes have a `get_llm_models` method. 
    # The tokenizer is inferred from the model name automatically. 
    llm_models = dummy_config.get_llm_models()
    if args.tokenizer_path is not None:
        tokenizer = get_tokenizer_for_model(args.tokenizer_path)
    else:
        tokenizer = None 

    for llm_model in llm_models:
        state = token_cost.get(llm_model)
        if state is not None:
            print(
                f"There is a saved checkpoint 📁 for monitoring the token consumption of {llm_model} 🤖. "
                "It will be loaded into `CostStateManager`."
            )
        CostStateManager.register(llm_model, state=state, tokenizer=tokenizer)
    del dummy_config 
    if len(llm_models) > 0:
        print(
            f"The LLM model(s) 🤖 being used are {', '.join(llm_models)}. " 
            "They have been registered in `CostStateManager`."
        )
        print()

    # Prepare the dataset using lazy mapping.
    ds_cls = DATASET_MAPPING[args.dataset_type]
    if args.dataset_standardized:
        dataset = ds_cls.read_dataset(args.dataset_path)
    else:
        dataset = ds_cls.read_raw_data(args.dataset_path) 
    if args.sample_size is not None or sample_filter is not None:
        dataset = dataset.sample(
            size=args.sample_size, 
            seed=args.seed,
            sample_filter=sample_filter,
        )
        dataset.save_dataset(f"{config['save_dir']}/{args.dataset_type}_stage_1")
    print("The dataset is loaded successfully 😄.")
    # `print(dataset)` calls the __str__ method defined by Pydantic's BaseModel.
    print(repr(dataset))
    print()

    if args.start_idx is None:
        args.start_idx = 0 
    if args.end_idx is None:
        args.end_idx = len(dataset)
    args.start_idx, args.end_idx = max(0, args.start_idx), min(args.end_idx, len(dataset))
    if args.start_idx >= args.end_idx:
        raise ValueError("The starting index must be less than the ending index.")

    results = [] 
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for trajectory, _ in zip(*dataset[args.start_idx: args.end_idx]):
            # Note that this code is for academic purpose, the embedding model will be loaded multiple times. 
            user_id = trajectory.id
            future = executor.submit(
                memory_construction, 
                args.memory_type, 
                user_id, 
                trajectory, 
                config=deepcopy(config), 
                rerun=args.rerun,
                message_preprocessor=message_preprocessor,
            )
            futures.append(future)
        for future in tqdm(
            as_completed(futures), 
            total=len(futures), 
            desc="📉 Processing trajectories"
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Error occurs when the trajectory is processed: {e}")

    if len(results) == args.end_idx - args.start_idx:
        print("The memory construction process is completed successfully 😀.")

    total_time = 0.0 
    avg_time_per_add_session = 0.0 
    num_vaild_trajectories = 0
    for result in results: 
        # Statistics on the newly processed trajectories
        if result["total_add_time"] > 0:
            total_time += result["total_add_time"]
            avg_time_per_add_session += result["avg_add_time"]
            num_vaild_trajectories += 1 
    avg_time = total_time / max(num_vaild_trajectories, 1)
    avg_time_per_add_session = avg_time_per_add_session / max(num_vaild_trajectories, 1)
    print(
        f"For {args.memory_type}, the average time per trajectory "
        f"({num_vaild_trajectories} in {len(results)}) is {avg_time:.2f} seconds."
    )
    print(
        f"For {args.memory_type}, the average time per operation of adding new session " 
        f"is {avg_time_per_add_session:.2f} seconds."
    )

    # Save the statistics of token comsumption 
    CostStateManager.save_to_json_file(args.token_cost_save_filename)
    CostStateManager.reset() 
