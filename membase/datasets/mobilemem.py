import json
from datetime import datetime, timedelta
from .base import MemBaseDataset
from ..model_types.dataset import (
    Trajectory,
    Session,
    QuestionAnswerPair,
    Message,
)
from typing import Any, Self


class MobileMem(MemBaseDataset):
    """Dataset wrapper for MobileMem."""

    @classmethod
    def read_raw_data(cls, path: str) -> Self:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        trajectories = []
        qa_pair_lists = []

        for sample in data:
            person = sample["person"]

            sessions = []
            for raw_session in sample["sessions"]:
                messages = []
                for raw_msg in raw_session["messages"]:
                    msg_metadata = raw_msg["metadata"].copy()
                    msg_metadata.update(
                        {
                            "side_note": raw_msg["side_note"],
                            "created_at": raw_msg["created_at"],
                        }
                    )

                    messages.append(
                        Message(
                            id=raw_msg["id"],
                            name=raw_msg["name"],
                            role=raw_msg["role"],
                            content=raw_msg["content"],
                            timestamp=raw_msg["timestamp"],
                            metadata=msg_metadata,
                        )
                    )

                session_metadata = {
                    key: value
                    for key, value in raw_session.items()
                    if key not in {"id", "messages"}
                }
                sessions.append(
                    Session(
                        id=raw_session["id"],
                        messages=messages,
                        metadata=session_metadata,
                    )
                )

            trajectories.append(
                Trajectory(
                    id=f"mobilemem-{person['id']}",
                    sessions=sorted(sessions),
                    metadata={
                        "person": person,
                        "graphs": sample["graphs"],
                    },
                )
            )

            question_ts = (
                datetime.fromisoformat(sessions[-1].ended_at) + timedelta(days=7)
            ).isoformat()

            qa_pairs = []
            qtoolbook = sample["question_type_toolbook"]
            for qtype in qtoolbook["question_types"]:
                for raw_qa in qtype["qa_pairs"]:
                    qa_pairs.append(
                        QuestionAnswerPair(
                            id=raw_qa["id"],
                            question=raw_qa["question"],
                            golden_answers=raw_qa["golden_answers"],
                            timestamp=question_ts,
                            metadata={
                                "question_type": raw_qa["question_type"],
                                "question_form": raw_qa["question_form"],
                                "difficulty": raw_qa["difficulty"],
                                "topic": raw_qa["topic"],
                                "num_hops": raw_qa["num_hops"],
                                "num_sub_questions": raw_qa["num_sub_questions"],
                                "effective_timestamp": raw_qa["effective_timestamp"],
                                "evidence": [
                                    ev["id"]
                                    for ev in raw_qa["source_evidences"]
                                ],
                                "is_consumed": raw_qa["is_consumed"],
                                "side_note": raw_qa["side_note"],
                                "created_at": raw_qa["created_at"],
                            },
                        )
                    )
            qa_pair_lists.append(qa_pairs)

        return cls(
            trajectories=trajectories,
            qa_pair_lists=qa_pair_lists,
        )

    def _generate_metadata(self) -> dict[str, Any]:
        meta = {
            "name": "MobileMem",
            "paper": "MobileMem: Evaluating Long-Horizon Memory for Language Agents in Real-World Mobile Environments",
            "codebase_url": "https://github.com/zjunlp/MobileMem",
            "size": len(self),
            "total_sessions": 0,
            "total_messages": 0,
            "total_questions": 0,
        }

        question_type_stats = {}
        difficulty_stats = {}
        question_form_stats = {}
        num_hops_stats = {}

        for trajectory, qa_list in self:
            meta["total_sessions"] += len(trajectory)
            meta["total_messages"] += sum(len(s) for s in trajectory)
            meta["total_questions"] += len(qa_list)
            for qa in qa_list:
                qtype = qa.metadata["question_type"]
                question_type_stats[qtype] = question_type_stats.get(qtype, 0) + 1
                difficulty = qa.metadata["difficulty"]
                difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
                form = qa.metadata["question_form"]
                question_form_stats[form] = question_form_stats.get(form, 0) + 1
                num_hops = str(qa.metadata["num_hops"])
                num_hops_stats[num_hops] = num_hops_stats.get(num_hops, 0) + 1

        meta["question_type_stats"] = question_type_stats
        meta["difficulty_stats"] = difficulty_stats
        meta["question_form_stats"] = question_form_stats
        meta["num_hops_stats"] = num_hops_stats

        n_traj = len(self)
        n_sessions = meta["total_sessions"]
        if n_traj > 0 and n_sessions > 0:
            meta["avg_session_per_trajectory"] = n_sessions / n_traj
            meta["avg_message_per_session"] = meta["total_messages"] / n_sessions
            meta["avg_question_per_trajectory"] = meta["total_questions"] / n_traj
        else:
            meta["avg_session_per_trajectory"] = 0.0
            meta["avg_message_per_session"] = 0.0
            meta["avg_question_per_trajectory"] = 0.0

        return meta

    @classmethod
    def get_judge_template_name(cls, qa_pair: QuestionAnswerPair) -> str:
        return "default-exact-match"
