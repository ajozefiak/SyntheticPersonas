from __future__ import annotations

import json
import random
from typing import Iterable, List, Sequence, Tuple

import dspy


def format_history(turns: Sequence[dict], question: str | None = None) -> str:
    """Format turns into a deterministic transcript history."""
    parts = []
    for turn in turns:
        turn_question = turn.get("q", "")
        answer = turn.get("a", "")
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

    for idx, interview in enumerate(interviews):
        persona_id = None
        if persona_list is not None:
            persona_id = persona_list[idx] if idx < len(persona_list) else None
        else:
            persona_id = str(idx)

        for turn_index, turn in enumerate(interview):
            history = format_history(interview[:turn_index])
            question = turn.get("q", "")
            answer = turn.get("a", "")
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
    """Split interviews and convert to DSPy train/val examples."""
    train_interviews, val_interviews = split_interviews(
        interviews, val_ratio=val_ratio, seed=seed
    )
    return build_examples(train_interviews), build_examples(val_interviews)


def load_interviews(path: str) -> List[List[dict]]:
    """Load interviews from JSON or JSONL.

    Expected format: list of interviews, each interview is a list of {"q", "a"}.
    """
    if path.endswith(".jsonl"):
        interviews: List[List[dict]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if isinstance(entry, dict) and "interview" in entry:
                    interviews.append(entry["interview"])
                else:
                    interviews.append(entry)
        return interviews

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "interviews" in data:
        return data["interviews"]
    return data
