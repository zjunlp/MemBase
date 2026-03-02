from __future__ import annotations
from datetime import datetime
from abc import ABC, abstractmethod
import random 
import uuid
from pydantic import (
    BaseModel, 
    Field, 
    field_validator,
    computed_field,
    PrivateAttr,
    ModelWrapValidatorHandler, 
    model_validator,
)
from ..inference_utils.operators import LLMExactMatch 
from typing import (
    Callable,
    Literal, 
    Any,
    Self, 
    Iterator, 
) 


class BaseMetadataModel(BaseModel):
    """Base class that provides a private metadata field and its serialization logic.
    
    This class ensures that metadata is stored as a private attribute while 
    remaining accessible and serializable through a computed field.
    """

    # Metadata of the object, used to store additional information.
    # It is marked as a private attribute to keep it out of the standard model fields.
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    def update_metadata(self, metadata: dict[str, Any]) -> None:
        """Update the metadata of the object.

        Args:
            metadata (`dict[str, Any]`):
                The metadata to be merged into the existing metadata.
        """
        self._metadata.update(metadata)

    @model_validator(mode="wrap")
    @classmethod
    def _restore_metadata_private_attrs(
        cls, 
        values: Any, 
        handler: ModelWrapValidatorHandler[Self]
    ) -> Self:
        """Restore private metadata from serialized data during deserialization.
        
        Args:
            values (`Any`):
                The input values to validate.
            handler (`ModelWrapValidatorHandler[Self]`):
                The handler function to create the instance.
        
        Returns:
            `Self`:
                The validated instance with private metadata restored.
        """
        if not isinstance(values, dict):
            return handler(values)
        
        # Extract metadata from the input dictionary if it exists.
        metadata = values.get("metadata", {}) 
        
        instance = handler(values)
        instance._metadata = metadata
        
        return instance

    @computed_field
    @property
    def metadata(self) -> dict[str, Any]:
        """Get the metadata of the object.
        
        Returns:
            `dict[str, Any]`:
                A copy of the metadata dictionary.
        """
        return self._metadata.copy()


class Message(BaseMetadataModel):
    """Represent a single message in a session."""

    id: str = Field(
        default_factory=lambda: f"message-{uuid.uuid4()}",
        description="Unique message identifier.",
    )
    name: str = Field(
        description="Name of the message sender.",
    )
    content: str = Field(
        description="Message content. Should be natural and contextually appropriate.",
    )
    role: Literal["user", "assistant", "system"] = Field(
        description=(
            "Role of the message sender. Must be one of: 'user', 'assistant', 'system'. "
            "'user' means the message is from the user, 'assistant' means the message is from the AI assistant, " 
            "'system' means the message is from the system which refers to an AI-centered integrated architecture " 
            "that encompasses the assistant, perception modules (e.g., sensors), external tools, memory components, " 
            "and actuators that collectively enable autonomous perception, reasoning, and action."
        ),
    )
    timestamp: str = Field(
        description=(
            "Timestamp when the message is sent, in ISO 8601 format."
        ),
    )

    def __lt__(self, other: Any) -> bool:
        """Compare messages based on their timestamp.

        Args:
            other (`Any`):
                The other message to compare with.

        Returns:
            `bool`:
                True if this message's timestamp is earlier than the other's.
        """
        if isinstance(other, Message | QuestionAnswerPair):
            return datetime.fromisoformat(self.timestamp) < datetime.fromisoformat(other.timestamp) 
        if isinstance(other, Session):
            return datetime.fromisoformat(self.timestamp) < datetime.fromisoformat(other.started_at)
        return NotImplemented

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate that `timestamp` is a valid ISO 8601 string."""
        try:
            _ = datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(
                f"The timestamp '{v}' is not in a valid format. "
                "Please use the format YYYY-MM-DD HH:MM:SS, for example: "
                "'2024-08-25 12:01:42'."
            )
        return v


class QuestionAnswerPair(BaseMetadataModel):
    """Represent a single question-answer pair for evaluation."""
    
    id: str = Field(
        default_factory=lambda: f"qa-{uuid.uuid4()}",
        description="Unique identifier for the question-answer pair.",
    )
    question: str = Field(
        description="The question to be answered.",
    )
    golden_answers: list[str] = Field(
        description=(
            "Reference answers. The answers should be direct natural language answers "
            "that do not reference any message IDs and session IDs."
        ),
        min_length=1,
    )
    timestamp: str = Field(
        description=(
            "Timestamp when the question is asked, in ISO 8601 format."
        ),
    )

    def __lt__(self, other: Any) -> bool:
        """Compare question-answer pairs based on their timestamp.

        Args:
            other (`Any`):
                The other question-answer pair to compare with.

        Returns:
            `bool`:
                True if this question-answer pair is asked earlier than the other's.
        """
        if isinstance(other, Message | QuestionAnswerPair):
            return datetime.fromisoformat(self.timestamp) < datetime.fromisoformat(other.timestamp) 
        if isinstance(other, Session):
            return datetime.fromisoformat(self.timestamp) < datetime.fromisoformat(other.started_at)
        return NotImplemented
    
    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate that `timestamp` is a valid ISO 8601 string."""
        try:
            _ = datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(
                f"The timestamp '{v}' is not in a valid format. "
                "Please use the format YYYY-MM-DD HH:MM:SS, for example: "
                "'2024-08-25 12:01:42'."
            )
        return v


class Session(BaseMetadataModel):
    """Represent a session in a trajectory."""

    id: str = Field(
        default_factory=lambda: f"session-{uuid.uuid4()}",
        description="Unique session identifier.",
    )
    messages: list[Message] = Field(
        description=(
            "Ordered list of messages in the session. Should form a "
            "coherent, natural session."
        ),
        min_length=1,
    )

    def __lt__(self, other: Any) -> bool:
        """Compare sessions based on their start time.

        Args:
            other (`Any`):
                The other session to compare with.

        Returns:
            `bool`:
                True if this session started earlier than the other's.
        """
        if isinstance(other, Session):
            return datetime.fromisoformat(self.started_at) < datetime.fromisoformat(other.started_at)
        if isinstance(other, Message | QuestionAnswerPair):
            return datetime.fromisoformat(self.started_at) < datetime.fromisoformat(other.timestamp)
        return NotImplemented
    
    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[Message]) -> list[Message]:
        """Validate that messages are in chronological order.
        
        Args:
            v (`list[Message]`):
                The list of messages in the session.
        
        Returns:
            `list[Message]`:
                The list of messages in the session.
        """
        prev_msg = None 
        for i, current_msg in enumerate(v): 
            if prev_msg is None: 
                prev_msg = current_msg
                continue
            if current_msg < prev_msg:
                raise ValueError(
                    "Messages must be in chronological order. "
                    f"The message at index {i} (timestamp: '{current_msg.timestamp}') "
                    f"is not later than the previous message at index {i - 1} (timestamp: '{v[i - 1].timestamp}'). "
                    "Please ensure all message timestamps are increasing."
                )
            prev_msg = current_msg
        return v

    @computed_field
    @property
    def started_at(self) -> str:
        """Return the start time of the session (the first message's timestamp) 
        in ISO 8601 format.
        
        Returns:
            `str`:
                The start time of the session in ISO 8601 format.
        """
        return self.messages[0].timestamp
    
    @computed_field
    @property
    def ended_at(self) -> str:
        """Return the end time of the session (the last message's timestamp) 
        in ISO 8601 format.
        
        Returns:
            `str`:
                The end time of the session in ISO 8601 format.
        """
        return self.messages[-1].timestamp

    @classmethod
    def create_from_messages(
        cls, 
        messages: list[Message], 
        **kwargs: Any
    ) -> Session:
        """Create a session instance by pre-sorting messages and setting metadata.
        
        Args:
            messages (`list[Message]`):
                A list of messages that might be unsorted.
            **kwargs (`Any`):
                Additional metadata to be stored in the session.
        
        Returns:
            `Session`:
                A new session instance with sorted messages and metadata.
        """
        # Pre-sort messages to satisfy the chronological validator. 
        # Note that we allow identical timestamps for now. 
        # In some datasets, there are multiple messages with the same timestamp.
        sorted_messages = sorted(messages)
        
        instance = cls(messages=sorted_messages)
        if kwargs:
            instance.update_metadata(kwargs)
            
        return instance
    
    def __len__(self) -> int:
        """Return the size of the session (i.e., the number of messages in the session)."""
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:
        """Iterate over the messages in the session."""
        return iter(self.messages)

    def __getitem__(self, index: int) -> Message:
        """Get the message at the given index."""
        return self.messages[index]


class Trajectory(BaseMetadataModel): 
    """Represent a user trajectory in the memory system evaluation datasets."""
    
    id: str = Field(
        default_factory=lambda: f"trajectory-{uuid.uuid4()}",
        description="Unique trajectory identifier.",
    )
    sessions: list[Session] = Field(
        ..., 
        description="The sessions in the trajectory.", 
        min_length=1,
    )

    @classmethod
    def create_from_sessions(
        cls, 
        sessions: list[Session], 
        **kwargs: Any
    ) -> Trajectory:
        """Create a trajectory instance by pre-sorting sessions and setting metadata.
        
        Args:
            sessions (`list[Session]`):
                A list of sessions that might be unsorted.
            **kwargs (`Any`):
                Additional metadata to be stored in the trajectory.
        
        Returns:
            `Trajectory`:
                A new trajectory instance with sorted sessions and metadata.
        """
        # Pre-sort sessions to satisfy the chronological validator. 
        # In some datasets, there are multiple sessions with the same start time.
        sorted_sessions = sorted(sessions)
        
        instance = cls(sessions=sorted_sessions)
        if kwargs:
            instance.update_metadata(kwargs)
            
        return instance

    def __len__(self) -> int:
        """Return the length of the user trajectory (i.e., the number of sessions in the trajectory)."""
        return len(self.sessions)

    def __iter__(self) -> Iterator[Session]:
        """Iterate over the sessions in the trajectory."""
        return iter(self.sessions)

    def __getitem__(self, index: int) -> Session:
        """Get the session at the given index."""
        return self.sessions[index]


class MemoryDataset(BaseMetadataModel, ABC):
    """A memory system evaluation dataset."""

    trajectories: list[Trajectory] = Field(
        ..., 
        description="The trajectories in the dataset.",
        min_length=1,
    )
    qa_pair_lists: list[list[QuestionAnswerPair]] = Field(
        ..., 
        description=(
            "The list of question-answer pairs for each trajectory in the dataset. "
            "The length of the list is the same as the number of trajectories."
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def _validate_lengths(self) -> Self:
        """Validate that the number of trajectories and the number of question-answer 
        pair lists are the same.
        
        Returns:
            `Self`:
                The validated instance.
        """
        if len(self.trajectories) != len(self.qa_pair_lists):
            raise ValueError(
                "The number of trajectories and the number of question-answer pair lists "
                "must be the same."
            )
        return self

    @abstractmethod
    def _generate_metadata(self) -> dict[str, Any]:
        """Generate the metadata of the dataset.

        This method should be overridden by subclasses to provide dataset-specific metadata.
        For example, some datasets may include question difficulty levels, while others 
        may provide source evidences, domain categories, or other annotations relevant 
        to the evaluation task.
        
        Returns:
            `dict[str, Any]`:
                The metadata of the dataset.
        """
        raise NotImplementedError("Subclasses should implement the method `_generate_metadata`.")

    @classmethod
    @abstractmethod
    def read_raw_data(cls, path: str) -> MemoryDataset:
        """Read the raw data from the given path and construct a dataset instance.

        Different datasets may have vastly different raw data formats (e.g., JSON,
        CSV, JSONL, or nested directory structures). Subclasses should implement this
        method to handle the specific data format and parsing logic of the target
        dataset, including loading raw files, constructing `Trajectory`, `Session`,
        `Message`, and `QuestionAnswerPair` objects, and assembling them into a
        complete `MemoryDataset`.

        Args:
            path (`str`):
                The path to the raw data file or directory.

        Returns:
            `MemoryDataset`:
                The dataset instance constructed from the raw data.
        """
        raise NotImplementedError("Subclasses should implement `read_raw_data`.")
    
    @model_validator(mode="after")
    def _process_metadata(self) -> Self:
        """Process the metadata of the dataset.
        
        Returns:
            `Self`:
                The dataset with the automatically generated metadata.
        """
        if not self._metadata:
            self._metadata = self._generate_metadata()
        return self

    def get_trajectories(self) -> list[Trajectory]:
        """Get the trajectories in the dataset.
        
        Returns:
            `list[Trajectory]`:
                The list of trajectories in the dataset.
        """
        return self.trajectories

    def get_qa_pair_lists(self) -> list[list[QuestionAnswerPair]]:
        """Get the question-answer pairs for each trajectory in the dataset.
        
        Returns:
            `list[list[QuestionAnswerPair]]`:
                The list of question-answer pairs for each trajectory in the dataset.
        """
        return self.qa_pair_lists
    
    def shuffle(self, seed: int | None = None) -> None:
        """Shuffle the dataset in-place.
        
        Args:
            seed (`int | None`):
                The seed for the random number generator.
        """
        rng = random.Random(seed)
        indices = list(range(len(self)))
        rng.shuffle(indices)
        self.trajectories = [self.trajectories[i] for i in indices]
        self.qa_pair_lists = [self.qa_pair_lists[i] for i in indices]

    def sample(
        self, 
        size: int | None = None, 
        seed: int | None = None,
        sample_filter: Callable[[Trajectory, list[QuestionAnswerPair]], bool] | None = None,
        question_filter: Callable[[QuestionAnswerPair], bool] | None = None,
    ) -> MemoryDataset:
        """Sample a subset from the dataset with optional filtering.

        This method supports a three-stage pipeline:

        1. Sample-level filtering: Each trajectory and its corresponding question-answer pair list are 
        evaluated by the filter predicate. Only samples for which the predicate returns `True` are retained. 
        This is useful, for example, to keep only trajectories whose token count is within a certain range.
        2. Question-level filtering: Within each retained sample, the question-answer pair list is further filtered 
        at the individual pair level. Samples whose question-answer pair list becomes empty after this step are discarded.
        This is useful, for example, to keep only question-answer pairs of a specific question type 
        (e.g., single-hop, multi-hop, temporal).
        3. Random sampling: A random subset of samples is drawn (without replacement) from the filtered results 
        based on the provided size parameter.

        If neither size parameter nor any filter is provided, a new dataset instance
        containing the same data is returned.

        Args:
            size (`int | None`, optional):
                The number of samples to draw after filtering. If it is not provided, all filtered
                samples are returned.
            seed (`int | None`, optional):
                The seed for the random number generator used in sampling.
            sample_filter (`Callable[[Trajectory, list[QuestionAnswerPair]], bool] | None`, optional):
                A predicate that takes a trajectory and its question-answer pair list, returning
                `True` to keep the sample. For example, filtering by trajectory token
                length::

                    lambda traj, qas: traj.metadata.get("num_tokens", 0) < 8000

            question_filter (`Callable[[QuestionAnswerPair], bool] | None`, optional):
                A predicate that takes a single question-answer pair, returning `True` to keep it.
                It is applied after the `sample_filter`. Samples with no remaining question-answer pairs
                are discarded. For example, keeping only single-hop questions::

                    lambda qa: qa.metadata.get("question_type") == "single-hop"

        Returns:
            `MemoryDataset`:
                A new dataset instance containing the filtered and sampled data.
        """
        trajectories = self.trajectories
        qa_pair_lists = self.qa_pair_lists

        # Stage 1: filter at the trajectory level.
        if sample_filter is not None:
            paired = [
                (traj, qas) 
                for traj, qas in zip(trajectories, qa_pair_lists)
                if sample_filter(traj, qas)
            ]
            trajectories = [t for t, _ in paired]
            qa_pair_lists = [q for _, q in paired]

        # Stage 2: filter at the individual question-answer pair level within each trajectory.
        if question_filter is not None:
            filtered_trajectories = []
            filtered_qa_pair_lists = []
            for traj, qas in zip(trajectories, qa_pair_lists):
                filtered_qas = [qa for qa in qas if question_filter(qa)]
                if filtered_qas:
                    filtered_trajectories.append(traj)
                    filtered_qa_pair_lists.append(filtered_qas)
            trajectories = filtered_trajectories
            qa_pair_lists = filtered_qa_pair_lists

        # Stage 3: random sampling from the (possibly filtered) results.
        if size is not None:
            if size > len(trajectories):
                raise ValueError(
                    f"It is impossible to sample {size} items from {len(trajectories)} available "
                    f"samples after filtering (the dataset originally has {len(self)} samples)."
                )
            rng = random.Random(seed)
            indices = rng.sample(range(len(trajectories)), size)
            trajectories = [trajectories[i] for i in indices]
            qa_pair_lists = [qa_pair_lists[i] for i in indices]

        return self.__class__(
            trajectories=trajectories,
            qa_pair_lists=qa_pair_lists,
        )
        
    def __repr__(self) -> str:
        """Return the string representation of the dataset."""
        def fmt_scalar(v: Any, width: int = 100) -> str:
            s = repr(v)
            return s if len(s) <= width else s[: width - 3] + "..."

        def render_dict(d: dict[str, Any], indent: int = 2, width: int = 100) -> list[str]:
            if not d:
                return []
            keys = sorted(map(str, d.keys()))
            key_w = max(len(k) for k in keys)
            lines = []
            for k in keys:
                v = d[k]
                # Use ljust to align the ':' column for all keys at this level
                pad = " " * indent + k.ljust(key_w) + ":"
                if isinstance(v, dict):
                    lines.append(pad)
                    lines.extend(render_dict(v, indent + 8, width))
                elif isinstance(v, (list, tuple, set)):
                    seq = list(v)
                    if not seq:
                        lines.append(pad + " []")
                    else:
                        lines.append(pad)
                        for x in seq:
                            if isinstance(x, dict):
                                lines.append(" " * (indent + 8) + "-")
                                lines.extend(render_dict(x, indent + 12, width))
                            else:
                                lines.append(" " * (indent + 8) + "- " + fmt_scalar(x, width))
                else:
                    lines.append(pad + " " + fmt_scalar(v, width))
            return lines

        header = f"{self.__class__.__name__} Metadata"
        bar = "─" * len(header)
        body_lines = render_dict(self.metadata, indent=2, width=100)

        return header + "\n" + bar + ("\n" + "\n".join(body_lines) if body_lines else "")
    
    @classmethod
    def get_judge_template_name(cls, qa_pair: QuestionAnswerPair) -> str:
        """Get the judge prompt template name for a question-answer pair.

        Subclasses can overwrite this method to customize the LLM-as-a-Judge prompt template.

        Args:
            qa_pair (`QuestionAnswerPair`):
                The question-answer pair to get the judge prompt template name for.

        Returns:
            `str`:
                The name of the judge prompt template.
        """
        return qa_pair.metadata.get("question_type", "default-exact-match")

    @classmethod
    def parse_judge_response(cls, content: str) -> float:
        """Convert the raw text output from the judge model into a correctness score.

        The default behaviour checks whether the word `"yes"` appears in the 
        lowercased response. Subclasses can override this method to accommodate 
        different judge formats.

        Args:
            content (`str`):
                The raw text content returned by the judge model.

        Returns:
            `float`:
                `1.0` if the prediction is judged correct, `0.0` otherwise.
        """
        return float("yes" in content.lower())

    @classmethod
    def evaluate(
        cls,
        qa_pairs: list[QuestionAnswerPair],
        predictions: list[str],
        judge_model: str = "gpt-4.1-mini",
        judge_batch_size: int = 4,
        **kwargs: Any,
    ) -> list[dict[str, float | str]]:
        """Evaluate the predictions against the golden answers for each question-answer pair.

        This base implementation uses an LLM-as-a-Judge approach to determine whether 
        each prediction is correct by comparing it against the golden answers. The judge 
        prompt template is resolved via ``get_judge_template_name``, and the raw judge 
        output is converted to a correctness score via ``parse_judge_response``. Subclasses 
        can override either of these class methods to customize the judging behaviour.

        Subclasses can also override this method to incorporate additional evaluation
        metrics beyond accuracy. For example, a dataset that annotates source evidence 
        in each question-answer pair could compute retrieval recall@k by comparing the 
        retrieved results against the ground-truth evidence IDs. When overriding, call 
        ``super().evaluate(...)`` first to obtain the base accuracy results, then merge 
        the extra per-pair metrics into each result dictionary::

            @classmethod
            def evaluate(cls, qa_pairs, predictions, **kwargs):
                results = super().evaluate(qa_pairs, predictions, **kwargs)
                retrieval_results = kwargs.get("retrieval_results", [])
                for i, qa_pair in enumerate(qa_pairs):
                    evidence_ids = set(qa_pair.metadata.get("evidence_ids", []))
                    retrieved_ids = set(r["id"] for r in retrieval_results[i])
                    hit = len(evidence_ids & retrieved_ids)
                    k = len(retrieval_results[i]) if retrieval_results else 1
                    results[i][f"recall@{k}"] = hit / len(evidence_ids) if evidence_ids else 0.0
                return results

        Args:
            qa_pairs (`list[QuestionAnswerPair]`):
                The question-answer pairs to evaluate.
            predictions (`list[str]`):
                The predicted answers, one per question-answer pair.
            judge_model (`str`, defaults to `"gpt-4.1-mini"`):
                The model name or path used for the LLM judge.
            judge_batch_size (`int`, defaults to `4`):
                Batch size for the judge model inference.
            **kwargs (`Any`):
                Remaining keyword arguments are forwarded to the LLM interface 
                constructor. If `api_key`, `api_keys`, `base_url` or `base_urls` is present, 
                an OpenAI-compatible API backend is used and the remaining arguments correspond 
                to those accepted by `openai.OpenAI`. Otherwise, a local vLLM backend is assumed and 
                the arguments correspond to `vllm.LLM` constructor parameters. An optional `generation_config` 
                dictionary can be included to supply generation-time parameters (mapping to the chat completions 
                request body for OpenAI, or `vllm.SamplingParams` for vLLM). If `generation_config` is not provided, 
                `{"temperature": 0.0}` is used by default for deterministic judging.

        Returns:
            `list[dict[str, float | str]]`:
                Per-pair evaluation results containing the accuracy and the judge response.
        """
        if len(qa_pairs) != len(predictions):
            raise ValueError(
                f"The number of question-answer pairs ({len(qa_pairs)}) and predictions "
                f"({len(predictions)}) must be the same."
            )

        # Separate generation-time config from interface constructor kwargs.
        generation_config = {"temperature": 0.0}
        generation_config.update(kwargs.pop("generation_config", {}))

        # Group question-answer pairs by their judge template name so that pairs sharing
        # the same prompt can be evaluated in a single batched call.
        groups = {}
        for idx, qa_pair in enumerate(qa_pairs):
            template_name = cls.get_judge_template_name(qa_pair)
            groups.setdefault(template_name, []).append(idx)

        results = [{} for _ in range(len(qa_pairs))]

        # Use the first group's template to initialize the operator; subsequent
        # groups switch the prompt via `set_prompt`.
        first_template = next(iter(groups))
        judge_operator = LLMExactMatch(
            prompt_name=first_template,
            model_name=judge_model,
            **kwargs,
        )

        for template_name, indices in groups.items():
            judge_operator.set_prompt(template_name)

            batch_questions = [qa_pairs[i].question for i in indices]
            batch_golden_answers = [qa_pairs[i].golden_answers for i in indices]
            batch_predictions = [predictions[i] for i in indices]

            judge_responses = judge_operator(
                batch_questions,
                batch_golden_answers,
                batch_predictions,
                batch_size=judge_batch_size,
                aggregate=False,
                **generation_config,
            )

            for local_pos, global_idx in enumerate(indices):
                content = judge_responses[local_pos].get("processed_content")
                if content is None:
                    raise ValueError(
                        "The judge model's response for question "
                        f"'{qa_pairs[global_idx].question}' is empty."
                    )
                results[global_idx] = {
                    "accuracy": cls.parse_judge_response(content),
                    "judge_response": content,
                }
            
            # Aggregate each group's results and print the average accuracy. 
            accuracy = sum([results[i]["accuracy"] for i in indices]) / len(indices)
            print(
                "The accuracy for the question-answer pairs evaluated with the judging template " 
                f"'{template_name}' is {accuracy:.4f}."
            )
        
        overall_accuracy = sum([results[i]["accuracy"] for i in range(len(results))]) / len(results)
        print(f"The overall accuracy is {overall_accuracy:.4f}.")

        return results

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.trajectories)

    def __iter__(self) -> Iterator[tuple[Trajectory, list[QuestionAnswerPair]]]:
        """Iterate over the trajectories and question-answer pair lists in the dataset."""
        return iter(zip(self.trajectories, self.qa_pair_lists))

    def __getitem__(self, index: int) -> tuple[Trajectory, list[QuestionAnswerPair]]:
        """Get the trajectory and question-answer pair list at the given index."""
        return self.trajectories[index], self.qa_pair_lists[index]