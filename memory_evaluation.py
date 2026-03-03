import argparse
from membase import (
    DATASET_MAPPING,
    EvaluationRunner,
    EvaluationRunnerConfig,
)
from membase.utils import import_function_from_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to evaluate the answers of the search results."
    )
    parser.add_argument(
        "--search-results-path",
        type=str,
        required=True,
        help="Path to the search results."
    )
    parser.add_argument(
        "--qa-model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name or path for question answering."
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4.1-mini",
        help="Model name or path for judgment (exact match)."
    )
    parser.add_argument(
        "--qa-batch-size",
        type=int,
        default=4,
        help="Batch size for question-answering."
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=4,
        help="Batch size for judgment."
    )
    parser.add_argument(
        "--api-config-path", 
        type=str, 
        default=None,
        help="Path to the API config file."
    )
    parser.add_argument(
        "--context-builder",
        type=str,
        default=None,
        help=(
            "Import path for a custom context builder function that converts a list of "
            "memory entries into a context string. "
            "It accepts 'module.submodule.function' or 'path/to/file.py:function'."
        ),
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help=(
            "Import path for a custom prompt template factory that returns a "
            "template with $question and $context placeholders. "
            "It accepts 'module.submodule.function' or 'path/to/file.py:function'."
        ),
    )
    parser.add_argument(
        "--add-question-timestamp",
        action="store_true",
        help="Append the question timestamp to the prompt.",
    )
    parser.add_argument(
        "--dataset-type", 
        choices=list(DATASET_MAPPING.keys()), 
        default=list(DATASET_MAPPING.keys())[0],
        type=str, 
        help="The type of the dataset used to evaluate the memory layer."
    )
    args = parser.parse_args()

    context_builder = (
        import_function_from_path(args.context_builder)
        if args.context_builder is not None else None
    )
    prompt_template = (
        import_function_from_path(args.prompt_template)
        if args.prompt_template is not None else None
    )

    runner_config = EvaluationRunnerConfig(
        search_results_path=args.search_results_path,
        dataset_type=args.dataset_type,
        qa_model=args.qa_model,
        judge_model=args.judge_model,
        qa_batch_size=args.qa_batch_size,
        judge_batch_size=args.judge_batch_size,
        api_config_path=args.api_config_path,
        context_builder=context_builder,
        prompt_template=prompt_template,
        add_question_timestamp=args.add_question_timestamp,
    )
    EvaluationRunner(runner_config).run()
