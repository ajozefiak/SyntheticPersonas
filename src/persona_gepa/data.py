from __future__ import annotations

import json
import random
from typing import Iterable, List, Sequence, Tuple

import dspy


def _extract_question_answer(turn: dict, context: str) -> Tuple[str, str]:
    if not isinstance(turn, dict):
        raise ValueError(f"{context} must be a dict with interview keys.")

    if "interviewer_question" in turn or "respondent_answer" in turn:
        if "interviewer_question" not in turn or "respondent_answer" not in turn:
            raise ValueError(
                f"{context} must include interviewer_question and respondent_answer."
            )
        return str(turn["interviewer_question"]), str(turn["respondent_answer"])

    if "q" in turn or "a" in turn:
        if "q" not in turn or "a" not in turn:
            raise ValueError(f"{context} must include q and a.")
        return str(turn["q"]), str(turn["a"])

    raise ValueError(
        f"{context} missing required keys (interviewer_question/respondent_answer or q/a)."
    )


def _normalize_interview(interview: Sequence[dict], context: str) -> List[dict]:
    if not isinstance(interview, list):
        raise ValueError(f"{context} must be a list of turns.")

    normalized: List[dict] = []
    for idx, turn in enumerate(interview):
        question, answer = _extract_question_answer(turn, f"{context} turn {idx}")
        normalized.append({"q": question, "a": answer})
    return normalized


def _coerce_interviews(data: object, context: str) -> List[List[dict]]:
    if isinstance(data, dict):
        if "interviews" in data:
            data = data["interviews"]
        elif "interview" in data:
            data = data["interview"]
        else:
            raise ValueError(f"{context} must be a list of interviews or turns.")

    if not isinstance(data, list) and isinstance(data, Sequence) and not isinstance(
        data, (str, bytes)
    ):
        data = list(data)

    if not isinstance(data, list):
        raise ValueError(f"{context} must be a list of interviews or turns.")

    if not data:
        return []

    if all(isinstance(item, dict) for item in data):
        return [data]

    if all(isinstance(item, list) for item in data):
        return data

    raise ValueError(f"{context} must be a list of interviews or turns.")


def format_history(turns: Sequence[dict], question: str | None = None) -> str:
    """Format turns into a deterministic transcript history."""
    parts = []
    for idx, turn in enumerate(turns):
        turn_question, answer = _extract_question_answer(turn, f"history turn {idx}")
        parts.append(f"Q: {turn_question}\nA: {answer}\n")
    if question is not None:
        parts.append(f"Q: {question}\n")
    return "".join(parts)


def build_examples(
    interviews: Sequence[Sequence[dict]],
    persona_ids: Iterable[str] | None = None,
) -> List[dspy.Example]:
    """Convert interview turns into DSPy Examples."""
    examples: List[dspy.Example] = []
    persona_list = list(persona_ids) if persona_ids is not None else None

    interview_list = _coerce_interviews(interviews, "interviews")
    normalized_interviews = [
        _normalize_interview(interview, f"interview {idx}")
        for idx, interview in enumerate(interview_list)
    ]

    for idx, interview in enumerate(normalized_interviews):
        persona_id = None
        if persona_list is not None:
            persona_id = persona_list[idx] if idx < len(persona_list) else None
        else:
            persona_id = str(idx)

        for turn_index, turn in enumerate(interview):
            history = format_history(interview[:turn_index])
            question = turn["q"]
            answer = turn["a"]
            example = dspy.Example(
                history=history,
                question=question,
                answer=answer,
                persona_id=persona_id,
            ).with_inputs("history", "question")
            examples.append(example)
    return examples


def split_interviews(
    interviews: Sequence[Sequence[dict]],
    val_ratio: float = 0.2,
    seed: int = 7,
) -> Tuple[List[Sequence[dict]], List[Sequence[dict]]]:
    """Split interviews into train/val lists by interview."""
    indices = list(range(len(interviews)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(len(indices) * val_ratio)) if interviews else 0
    val_indices = set(indices[:val_count])
    train, val = [], []
    for idx, interview in enumerate(interviews):
        if idx in val_indices:
            val.append(interview)
        else:
            train.append(interview)
    return train, val


def build_train_val_examples(
    interviews: Sequence[Sequence[dict]],
    val_ratio: float = 0.2,
    seed: int = 7,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Split interviews temporally and convert to DSPy train/val examples."""
    if val_ratio < 0 or val_ratio >= 1:
        raise ValueError("val_ratio must be in [0, 1).")

    train_examples: List[dspy.Example] = []
    val_examples: List[dspy.Example] = []

    interview_list = _coerce_interviews(interviews, "interviews")
    normalized_interviews = [
        _normalize_interview(interview, f"interview {idx}")
        for idx, interview in enumerate(interview_list)
    ]

    for idx, interview in enumerate(normalized_interviews):
        if not interview:
            continue
        total_turns = len(interview)
        split_idx = max(1, int(total_turns * (1 - val_ratio)))
        if total_turns > 1 and split_idx >= total_turns:
            split_idx = total_turns - 1

        persona_id = str(idx)
        for turn_index, turn in enumerate(interview):
            history = format_history(interview[:turn_index])
            example = dspy.Example(
                history=history,
                question=turn["q"],
                answer=turn["a"],
                persona_id=persona_id,
            ).with_inputs("history", "question")
            if turn_index < split_idx:
                train_examples.append(example)
            else:
                val_examples.append(example)

    return train_examples, val_examples


def load_interviews(path: str) -> List[List[dict]]:
    """Load interviews from JSON or JSONL."""
    if path.endswith(".jsonl"):
        interviews: List[List[dict]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if isinstance(entry, dict) and "interview" in entry:
                    entry = entry["interview"]
                if not isinstance(entry, list):
                    raise ValueError(
                        f"{path}:{line_number} must be a list of interview turns."
                    )
                interviews.append(entry)
        return [
            _normalize_interview(interview, f"{path} line {idx}")
            for idx, interview in enumerate(interviews, start=1)
        ]

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    interview_list = _coerce_interviews(data, path)
    return [
        _normalize_interview(interview, f"{path} interview {idx}")
        for idx, interview in enumerate(interview_list)
    ]
