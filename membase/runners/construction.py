import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from pydantic import (
    BaseModel, 
    ConfigDict, 
    Field,
)
from tqdm import tqdm
from ..configs import CONFIG_MAPPING
from ..datasets import DATASET_MAPPING
from ..layers import MEMORY_LAYERS_MAPPING
from ..model_types.dataset import (
    Message, 
    Trajectory, 
    QuestionAnswerPair,
)
from ..utils import (
    CostState,
    CostStateManager,
    MonkeyPatcher,
    get_tokenizer_for_model, 
)
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
    """Given a specific interaction trajectory, build a memory for one user.

    This function is designed to be submitted to a thread pool so that
    multiple trajectories can be processed in parallel.

    Args:
        layer_type (`str`): 
            The registered name of the memory layer.
        user_id (`str`): 
            Unique identifier of the user whose memory is being built.
        trajectory (`Trajectory`): 
            The interaction trajectory consisting of sessions and messages.
        config (`dict[str, Any] | None`, optional): 
            Raw configuration dictionary for the memory layer. If not provided, 
            defaults will be used.
        rerun (`bool`, defaults to `False`): 
            If `False`, skip construction when a saved memory already exists.
        message_preprocessor (`Callable[[Message], Message]`, optional): 
            A callable that preprocesses each message before it is added to the memory.

    Returns:
        `dict[str, float]`: 
            A dictionary containing the total time and the average time per operation of 
            adding new message.
    """
    config = deepcopy(config) or {}
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


class ConstructionRunnerConfig(BaseModel):
    """Configuration for the construction runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_type: str = Field(
        ..., 
        description="The type of the memory layer to be evaluated.",
    )
    dataset_type: str = Field(
        ..., 
        description="The type of the dataset used to evaluate the memory layer.",
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
        description="Path to the JSON config for the memory method.",
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
        description="The number of threads to use for the evaluation.",
    )
    seed: int = Field(
        default=42,
        description=(
            "Random seed used to sample the dataset if the user provides the sample size."
        ),
    )
    sample_size: int | None = Field(
        default=None,
        description="Subset size from dataset.",
    )
    rerun: bool = Field(
        default=False,
        description="Ignore saved memory and rebuild the memory from scratch.",
    )
    token_cost_save_filename: str = Field(
        default="token_cost",
        description="Path to save the statistics related to the token consumption.",
    )
    start_idx: int | None = Field(
        default=None,
        description="The starting index of the trajectories to be processed.",
    )
    end_idx: int | None = Field(
        default=None,
        description="The ending index of the trajectories to be processed.",
    )
    tokenizer_path: str | None = Field(
        default=None,
        description="The path to the tokenizer (only for backbone model).",
    )
    message_preprocessor: Callable[[Message], Message] | None = Field(
        default=None,
        description=(
            "A callable that preprocesses each message before it is added to the memory."
        ),
    )
    sample_filter: Callable[[Trajectory, list[QuestionAnswerPair]], bool] | None = Field(
        default=None,
        description="A callable that filters dataset samples.",
    )


class ConstructionRunner:
    """Runner that orchestrates the memory construction stage.

    It loads the dataset, sets up token-cost monitoring, and dispatches 
    per-trajectory construction tasks to a thread pool.
    """

    def __init__(self, config: ConstructionRunnerConfig) -> None:
        """Initialize the construction runner.

        Args:
            config (`ConstructionRunnerConfig`): 
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

    def run(self) -> list[dict[str, float]]:
        """Execute the memory construction pipeline.

        Returns:
            `list[dict[str, float]]`: 
                A list of per-trajectory timing dictionaries, each containing 
                the total time and the average time per operation of adding new message.
        """
        cfg = self.config
        config = self._resolve_memory_config()

        # Get a dummy configuration to infer the corresponding LLM being used.
        # Use lazy mapping to load config class.
        config_cls = CONFIG_MAPPING[cfg.memory_type]
        if config is None:
            dummy_config = config_cls(user_id="guest")
        else:
            dummy_config = config_cls(**config)

        # If token cost file exists, load it.
        if os.path.exists(cfg.token_cost_save_filename + ".json"):
            with open(cfg.token_cost_save_filename + ".json", "r") as f:
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
        if cfg.tokenizer_path is not None:
            tokenizer = get_tokenizer_for_model(cfg.tokenizer_path)
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
        ds_cls = DATASET_MAPPING[cfg.dataset_type]
        if cfg.dataset_standardized:
            dataset = ds_cls.read_dataset(cfg.dataset_path)
        else:
            dataset = ds_cls.read_raw_data(cfg.dataset_path)
        if cfg.sample_size is not None or cfg.sample_filter is not None:
            dataset = dataset.sample(
                size=cfg.sample_size,
                seed=cfg.seed,
                sample_filter=cfg.sample_filter,
            )
            dataset.save_dataset(f"{config['save_dir']}/{cfg.dataset_type}_stage_1")
        print("The dataset is loaded successfully 😄.")
        # `print(dataset)` calls the __str__ method defined by Pydantic's BaseModel.
        print(repr(dataset))
        print()

        # Resolve index range.
        start_idx = cfg.start_idx if cfg.start_idx is not None else 0
        end_idx = cfg.end_idx if cfg.end_idx is not None else len(dataset)
        start_idx = max(0, start_idx)
        end_idx = min(end_idx, len(dataset))
        if start_idx >= end_idx:
            raise ValueError("The starting index must be less than the ending index.")

        # Dispatch per-trajectory construction to a thread pool.
        results = []
        with ThreadPoolExecutor(max_workers=cfg.num_workers) as executor:
            futures = []
            for trajectory, _ in zip(*dataset[start_idx:end_idx]):
                # Note that this code is for academic purpose, the embedding model may be loaded multiple times. 
                user_id = trajectory.id
                future = executor.submit(
                    memory_construction,
                    cfg.memory_type,
                    user_id,
                    trajectory,
                    config=config,
                    rerun=cfg.rerun,
                    message_preprocessor=cfg.message_preprocessor,
                )
                futures.append(future)
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="📉 Processing trajectories",
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"❌ Error occurs when the trajectory is processed: {e}")

        if len(results) == end_idx - start_idx:
            print("The memory construction process is completed successfully 😀.")

        # Print statistics.
        total_time = 0.0
        avg_time_per_add_session = 0.0
        num_valid_trajectories = 0
        for result in results:
            # Statistics on the newly processed trajectories. 
            if result["total_add_time"] > 0:
                total_time += result["total_add_time"]
                avg_time_per_add_session += result["avg_add_time"]
                num_valid_trajectories += 1
        avg_time = total_time / max(num_valid_trajectories, 1)
        avg_time_per_add_session = avg_time_per_add_session / max(num_valid_trajectories, 1)
        print(
            f"For {cfg.memory_type}, the average time per trajectory "
            f"({num_valid_trajectories} in {len(results)}) is {avg_time:.2f} seconds."
        )
        print(
            f"For {cfg.memory_type}, the average time per operation of adding new session "
            f"is {avg_time_per_add_session:.2f} seconds."
        )

        # Save token cost statistics.
        CostStateManager.save_to_json_file(cfg.token_cost_save_filename)
        CostStateManager.reset()

        return results
