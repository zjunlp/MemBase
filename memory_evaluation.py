from string import Template
from membase.model_types.dataset import QuestionAnswerPair
from membase.inference_utils.operators import QuestionAnsweringOperator
from membase.model_types.memory import MemoryEntry
from membase.utils.files import import_function_from_path
from membase import DATASET_MAPPING
import argparse
import json 
import os 
from typing import Any, Callable


def answer_questions(
    retrievals: list[dict[str, Any]],
    qa_model: str,
    qa_batch_size: int = 4,
    add_question_timestamp: bool = False, 
    prompt_template: Callable[[], Template] | None = None,
    context_builder: Callable[[list[MemoryEntry]], str] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Answer questions using retrieved memories and an LLM."""
    interface_kwargs = interface_kwargs or {}

    if context_builder is None:
        context_builder = lambda memories: "\n\n".join(
            f"### Memory {i + 1}:\n{mem.formatted_content or mem.content}"
            for i, mem in enumerate(memories)
        )

    questions = []
    contexts = []
    for item in retrievals:
        qa_pair = item["qa_pair"]
        question = qa_pair.question
        if "name" in qa_pair.metadata:
            question = f"{qa_pair.metadata['name']}: {question}"
        if add_question_timestamp:
            questions.append(
                f"{question}\nQuestion Timestamp: {qa_pair.timestamp}"
            )
        else:
            questions.append(question)
        contexts.append(context_builder(item["retrieved_memories"]))

    qa_operator = QuestionAnsweringOperator(
        prompt_name="default-question-answering",
        model_name=qa_model,
        timeout=120.0, 
        **interface_kwargs,
    )

    if prompt_template is not None:
        qa_operator.set_prompt(prompt_template())

    responses = qa_operator(
        questions,
        contexts,
        batch_size=qa_batch_size,
        aggregate=False,
        temperature=0.0,
    )
    return responses


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

    # Prepare interface key-value pairs.
    interface_kwargs = {}
    if args.api_config_path is not None:
        with open(args.api_config_path, 'r') as f:
            api_config = json.load(f)
        interface_kwargs["api_keys"] = api_config["api_keys"]
        interface_kwargs["base_urls"] = api_config["base_urls"]
    elif os.environ.get("OPENAI_API_KEY") is not None:
        interface_kwargs["api_keys"] = [os.environ.get("OPENAI_API_KEY")]
        interface_kwargs["base_urls"] = [os.environ.get("OPENAI_API_BASE")]
    
    # Resolve custom components from import paths.
    context_builder = (
        import_function_from_path(args.context_builder)
        if args.context_builder is not None else None
    )
    prompt_template = (
        import_function_from_path(args.prompt_template)
        if args.prompt_template is not None else None
    )

    dataset_cls = DATASET_MAPPING[args.dataset_type]

    with open(args.search_results_path, 'r') as f:
        retrievals = json.load(f)
    for item in retrievals:
        item["qa_pair"] = QuestionAnswerPair(**item["qa_pair"])
        item["retrieved_memories"] = [
            MemoryEntry(**mem) for mem in item["retrieved_memories"]
        ]
    print(f"✅ {len(retrievals)} retrieval results are loaded from {args.search_results_path}.")

    print("🧠 Generating answers...")
    qa_responses = answer_questions(
        retrievals,
        qa_model=args.qa_model,
        qa_batch_size=args.qa_batch_size,
        add_question_timestamp=args.add_question_timestamp,
        prompt_template=prompt_template,
        context_builder=context_builder,
        interface_kwargs=interface_kwargs,
    )

    # Extract prediction strings from raw LLM responses. 
    predictions = []
    for resp in qa_responses:
        pred = resp.get("processed_content")
        if pred is None:
            raise ValueError("The question-answering model returns an empty prediction.")
        predictions.append(pred)

    # Evaluate answers via the dataset class's judge logic.
    print("⚖️ Evaluating answers...")
    qa_pairs = [item["qa_pair"] for item in retrievals]
    judge_results = dataset_cls.evaluate(
        qa_pairs=qa_pairs,
        predictions=predictions,
        judge_model=args.judge_model,
        judge_batch_size=args.judge_batch_size,
        **interface_kwargs,
    )

    # Assemble final outputs.
    final_results = []
    for i, item in enumerate(retrievals):
        qa_pair = item["qa_pair"]
        final_results.append(
            {
                "qa_pair": qa_pair.model_dump(mode="python"),
                "prediction": predictions[i],
                "accuracy": judge_results[i]["accuracy"],
                "judge_response": judge_results[i]["judge_response"],
                "retrieved_memories": [
                    mem.model_dump(mode="python") for mem in item["retrieved_memories"]
                ],
                "user_id": item["user_id"],
            }
        )

    output_path = args.search_results_path.rsplit('.', 1)[0] + "_evaluation.json"
    with open(
        output_path, 
        'w', 
        encoding="utf-8"
    ) as f:
        json.dump(
            final_results, 
            f, 
            ensure_ascii=False, 
            indent=4, 
        )
    print(f"✅ {len(final_results)} evaluation results are saved to {output_path}.")