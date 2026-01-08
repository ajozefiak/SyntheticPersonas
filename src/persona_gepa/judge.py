from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import ClassVar, Dict

import dspy


JUDGE_INSTRUCTIONS = (
    "You are a strict evaluator of interview answers. "
    "Score the candidate answer versus the reference answer and transcript history. "
    "Return ONLY a JSON object with keys: accuracy, faithfulness, tone, style, "
    "feedback. Each score must be a float in [0,1]. Feedback must be a short, "
    "actionable string."
)


@dataclass
class Judgment:
    accuracy: float
    faithfulness: float
    tone: float
    style: float
    feedback: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "faithfulness": self.faithfulness,
            "tone": self.tone,
            "style": self.style,
            "feedback": self.feedback,
        }


class JudgeSignature(dspy.Signature):
    """You are a strict evaluator of interview answers. Score the candidate answer versus the reference answer and transcript history. Return ONLY a JSON object with keys: accuracy, faithfulness, tone, style, feedback. Each score must be a float in [0,1]. Feedback must be a short, actionable string."""

    instructions: ClassVar[str] = JUDGE_INSTRUCTIONS

    history = dspy.InputField(desc="Transcript history.")
    question = dspy.InputField(desc="Current question.")
    reference_answer = dspy.InputField(desc="Ground truth answer.")
    candidate_answer = dspy.InputField(desc="Model-generated answer.")

    judgment = dspy.OutputField(desc="Strict JSON string with scores and feedback.")


class JudgeProgram(dspy.Module):
    def __init__(self, lm=None):
        super().__init__()
        self.predict = dspy.Predict(JudgeSignature, lm=lm)

    def forward(
        self,
        history: str,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ):
        return self.predict(
            history=history,
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_judgment(payload: Dict[str, object]) -> Judgment:
    def _get_score(key: str) -> float:
        raw = payload.get(key, 0.0)
        try:
            return _clamp(float(raw))
        except (TypeError, ValueError):
            return 0.0

    feedback = payload.get("feedback") or "No feedback provided."
    return Judgment(
        accuracy=_get_score("accuracy"),
        faithfulness=_get_score("faithfulness"),
        tone=_get_score("tone"),
        style=_get_score("style"),
        feedback=str(feedback),
    )


def _extract_json_blob(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None


def parse_judge_output(raw_output: object) -> Judgment:
    if isinstance(raw_output, Judgment):
        return raw_output

    if isinstance(raw_output, dict):
        return _normalize_judgment(raw_output)

    text = str(raw_output or "").strip()
    if not text:
        return Judgment(0.0, 0.0, 0.0, 0.0, "No judgment returned.")

    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return _normalize_judgment(payload)
    except json.JSONDecodeError:
        pass

    blob = _extract_json_blob(text)
    if blob:
        try:
            payload = json.loads(blob)
            if isinstance(payload, dict):
                return _normalize_judgment(payload)
        except json.JSONDecodeError:
            pass

    payload: Dict[str, object] = {}
    for key in ("accuracy", "faithfulness", "tone", "style"):
        match = re.search(rf"{key}\s*[:=]\s*([0-9]*\.?[0-9]+)", text)
        if match:
            payload[key] = match.group(1)
    feedback_match = re.search(r"feedback\s*[:=]\s*(.*)", text)
    if feedback_match:
        payload["feedback"] = feedback_match.group(1).strip()

    if payload:
        return _normalize_judgment(payload)

    return Judgment(0.0, 0.0, 0.0, 0.0, "Failed to parse judge output.")
