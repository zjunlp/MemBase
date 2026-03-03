import json
import os
from string import Template
from pydantic import (
    BaseModel, 
    ConfigDict, 
    Field,
)
from ..datasets import DATASET_MAPPING
from ..inference_utils.operators import QuestionAnsweringOperator
from ..model_types.dataset import QuestionAnswerPair
from ..model_types.memory import MemoryEntry
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
    """Answer questions using retrieved memories and an LLM.

    Args:
        retrievals (`list[dict[str, Any]]`): 
            The retrieval results produced by the search runner.
        qa_model (`str`): 
            Model name or path for question answering.
        qa_batch_size (`int`, defaults to `4`): 
            Batch size for question-answering.
        add_question_timestamp (`bool`, defaults to `False`): 
            Whether to append the question timestamp to the prompt.
        prompt_template (`Callable[[], Template] | None`, optional): 
            A factory that returns a `string.Template` with 
            `$question` and `$context` placeholders.
        context_builder (`Callable[[list[MemoryEntry]], str] | None`, optional): 
            A callable that converts a list of memory entries into a single 
            context string.
        interface_kwargs (`dict[str, Any] | None`, optional): 
            Extra keyword arguments forwarded to the LLM operator.

    Returns:
        `list[dict[str, Any]]`: 
            Raw LLM response dictionaries.
    """
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


class EvaluationRunnerConfig(BaseModel):
    """Configuration for the evaluation runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    search_results_path: str = Field(
        ...,
        description="Path to the search results.",
    )
    dataset_type: str = Field(
        ...,
        description="The type of the dataset used to evaluate the memory layer.",
    )
    qa_model: str = Field(
        default="gpt-4.1-mini",
        description="Model name or path for question answering.",
    )
    judge_model: str = Field(
        default="gpt-4.1-mini",
        description="Model name or path for judgment.",
    )
    qa_batch_size: int = Field(
        default=4,
        description="Batch size for question-answering.",
    )
    judge_batch_size: int = Field(
        default=4,
        description="Batch size for judgment.",
    )
    api_config_path: str | None = Field(
        default=None,
        description="Path to the API config file.",
    )
    api_keys: list[str] | None = Field(
        default=None,
        description=(
            "API keys for the LLM operator. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )
    base_urls: list[str] | None = Field(
        default=None,
        description=(
            "Base URLs for the LLM operator. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )
    context_builder: Callable[[list[MemoryEntry]], str] | None = Field(
        default=None,
        description=(
            "A callable that converts a list of memory entries into a context string."
        ),
    )
    prompt_template: Callable[[], Template] | None = Field(
        default=None,
        description=(
            "A factory that returns a ``string.Template`` with "
            "``$question`` and ``$context`` placeholders."
        ),
    )
    add_question_timestamp: bool = Field(
        default=False,
        description="Append the question timestamp to the prompt.",
    )


class EvaluationRunner:
    """Runner that orchestrates the question-answering and evaluation stage.

    It loads retrieval results, generates answers via an LLM, and then 
    delegates judgment to the dataset-specific evaluation logic.
    """

    def __init__(self, config: EvaluationRunnerConfig) -> None:
        """Initialize the evaluation runner.

        Args:
            config (`EvaluationRunnerConfig`): 
                The runner configuration.
        """
        self.config = config

    def _resolve_interface_kwargs(self) -> dict[str, Any]:
        """Build the interface keyword arguments for the LLM operator."""
        cfg = self.config
        interface_kwargs = {}

        if cfg.api_keys is not None and cfg.base_urls is not None:
            interface_kwargs["api_keys"] = cfg.api_keys
            interface_kwargs["base_urls"] = cfg.base_urls
        elif cfg.api_config_path is not None:
            with open(cfg.api_config_path, "r") as f:
                api_config = json.load(f)
            interface_kwargs["api_keys"] = api_config["api_keys"]
            interface_kwargs["base_urls"] = api_config["base_urls"]
        elif os.environ.get("OPENAI_API_KEY") is not None:
            interface_kwargs["api_keys"] = [os.environ["OPENAI_API_KEY"]]
            interface_kwargs["base_urls"] = [os.environ.get("OPENAI_API_BASE")]

        return interface_kwargs

    def run(self) -> list[dict[str, Any]]:
        """Execute the question-answering and evaluation pipeline.

        Returns:
            `list[dict[str, Any]]`: 
                A list of evaluation results. Each element is a dictionary 
                containing the question-answer pair, the prediction, the accuracy, 
                the judge response, the retrieved memories, and the user id.
        """
        cfg = self.config
        interface_kwargs = self._resolve_interface_kwargs()
        dataset_cls = DATASET_MAPPING[cfg.dataset_type]

        # Load and deserialize retrieval results.
        with open(cfg.search_results_path, "r") as f:
            retrievals = json.load(f)
        for item in retrievals:
            item["qa_pair"] = QuestionAnswerPair(**item["qa_pair"])
            item["retrieved_memories"] = [
                MemoryEntry(**mem) for mem in item["retrieved_memories"]
            ]
        print(
            f"✅ {len(retrievals)} retrieval results are loaded "
            f"from {cfg.search_results_path}."
        )

        # Generate answers.
        print("🧠 Generating answers...")
        qa_responses = answer_questions(
            retrievals,
            qa_model=cfg.qa_model,
            qa_batch_size=cfg.qa_batch_size,
            add_question_timestamp=cfg.add_question_timestamp,
            prompt_template=cfg.prompt_template,
            context_builder=cfg.context_builder,
            interface_kwargs=interface_kwargs,
        )

        # Extract prediction strings from raw LLM responses.
        predictions = []
        for resp in qa_responses:
            pred = resp.get("processed_content")
            if pred is None:
                raise ValueError(
                    "The question-answering model returns an empty prediction."
                )
            predictions.append(pred)

        # Evaluate answers via the dataset class's judge logic.
        print("⚖️ Evaluating answers...")
        qa_pairs = [item["qa_pair"] for item in retrievals]
        judge_results = dataset_cls.evaluate(
            qa_pairs=qa_pairs,
            predictions=predictions,
            judge_model=cfg.judge_model,
            judge_batch_size=cfg.judge_batch_size,
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
                        mem.model_dump(mode="python")
                        for mem in item["retrieved_memories"]
                    ],
                    "user_id": item["user_id"],
                }
            )

        # Persist results.
        output_path = (
            cfg.search_results_path.rsplit(".", 1)[0] + "_evaluation.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                final_results,
                f,
                ensure_ascii=False,
                indent=4,
            )
        print(f"✅ {len(final_results)} evaluation results are saved to {output_path}.")

        return final_results
