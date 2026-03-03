import argparse
from membase import (
    MEMORY_LAYERS_MAPPING,
    DATASET_MAPPING,
    ConstructionRunner,
    ConstructionRunnerConfig,
)
from membase.utils import import_function_from_path


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

    runner_config = ConstructionRunnerConfig(
        memory_type=args.memory_type,
        dataset_type=args.dataset_type,
        dataset_path=args.dataset_path,
        dataset_standardized=args.dataset_standardized,
        config_path=args.config_path,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_size=args.sample_size,
        rerun=args.rerun,
        token_cost_save_filename=args.token_cost_save_filename,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        tokenizer_path=args.tokenizer_path,
        message_preprocessor=message_preprocessor,
        sample_filter=sample_filter,
    )
    ConstructionRunner(runner_config).run()
