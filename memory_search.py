import argparse
from membase.utils import import_function_from_path
from membase import (
    MEMORY_LAYERS_MAPPING,
    DATASET_MAPPING,
    SearchRunner,
    SearchRunnerConfig,
)


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
        "--no-strict",
        action="store_true",
        help=(
            "Disable strict mode. When set, missing memories for a user are "
            "skipped instead of raising an error."
        ),
    )
    parser.add_argument(
        "--traced-data-save-dir",
        type=str,
        default="traced_data",
        help="Directory where execution graph artefacts are saved."
    )
    parser.add_argument(
        "--tracing",
        action="store_true",
        help="Enable execution graph tracing.",
    )
    args = parser.parse_args()

    question_filter = None
    if args.question_filter_path is not None:
        question_filter = import_function_from_path(args.question_filter_path)

    runner_config = SearchRunnerConfig(
        memory_type=args.memory_type,
        dataset_type=args.dataset_type,
        dataset_path=args.dataset_path,
        dataset_standardized=args.dataset_standardized,
        config_path=args.config_path,
        num_workers=args.num_workers,
        top_k=args.top_k,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        strict=not args.no_strict,
        question_filter=question_filter,
        traced_data_save_dir=args.traced_data_save_dir,
        tracing=args.tracing,
    )
    SearchRunner(runner_config).run()
