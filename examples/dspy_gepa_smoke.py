"""Minimal DSPy + GEPA smoke test for step-by-step debugging in notebooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, List

import dspy


# --- Configuration helpers ---


def _get_api_base() -> str | None:
    return os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


# --- Simple dataset ---


@dataclass
class QAExample:
    history: str
    question: str
    answer: str


def build_dataset() -> List[QAExample]:
    return [
        QAExample(
            history="Q: Where did you grow up?\nA: I grew up in Austin.",
            question="What do you enjoy doing?",
            answer="I like hiking and reading.",
        ),
        QAExample(
            history="Q: What is your job?\nA: I write scripts for professors.",
            question="What do you do for work?",
            answer="I write scripts for professors.",
        ),
    ]


# --- Program under optimization ---


class PersonaSignature(dspy.Signature):
    """Answer as the interviewee. Stay faithful to the transcript history."""

    instructions: ClassVar[str] = (
        "Answer as the interviewee. Be faithful to the transcript history."
    )

    history = dspy.InputField(desc="Transcript history.")
    question = dspy.InputField(desc="Current question.")
    answer = dspy.OutputField(desc="Interviewee-style answer.")


class PersonaProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(PersonaSignature)

    def forward(self, history: str, question: str):
        return self.predict(history=history, question=question)


# --- Optional judge (LLM-based) ---


class JudgeSignature(dspy.Signature):
    """Return JSON with keys: score (0-1 float), feedback (short string)."""

    instructions: ClassVar[str] = (
        "Score the candidate answer vs the reference answer. "
        "Return ONLY JSON with keys: score, feedback."
    )

    history = dspy.InputField(desc="Transcript history.")
    question = dspy.InputField(desc="Current question.")
    reference_answer = dspy.InputField(desc="Ground truth answer.")
    candidate_answer = dspy.InputField(desc="Model answer.")
    judgment = dspy.OutputField(desc="JSON: {\"score\": 0-1, \"feedback\": \"...\"}")


class JudgeProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(JudgeSignature)

    def forward(self, history: str, question: str, reference_answer: str, candidate_answer: str):
        return self.predict(
            history=history,
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )


def build_metric(judge: JudgeProgram, judge_lm):
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        candidate = getattr(pred, "answer", str(pred))
        with dspy.context(lm=judge_lm):
            judge_pred = judge(
                history=gold.history,
                question=gold.question,
                reference_answer=gold.answer,
                candidate_answer=candidate,
            )
        raw = getattr(judge_pred, "judgment", "{}")
        try:
            data = dspy.parse_json(raw)
            score = float(data.get("score", 0.0))
            feedback = str(data.get("feedback", ""))
        except Exception:
            score = 0.0
            feedback = "Failed to parse judge output."
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def main():
    # 1) Set these env vars in your notebook before calling main():
    # os.environ["OPENAI_API_KEY"] = "..."
    # os.environ["OPENAI_API_BASE"] = "https://your-gateway.example.com/api/v2"
    api_key = _require_env("OPENAI_API_KEY")
    api_base = _get_api_base()

    print("OPENAI_API_BASE =", api_base)
    print("Model =", "openai/gpt-4o")

    # 2) Build LMs (persona, judge, reflection)
    persona_lm = dspy.LM(
        model="openai/gpt-4o",
        api_key=api_key,
        api_base=api_base,
        temperature=0.2,
        max_tokens=256,
    )
    judge_lm = dspy.LM(
        model="openai/gpt-4o",
        api_key=api_key,
        api_base=api_base,
        temperature=0.0,
        max_tokens=256,
    )
    reflection_lm = dspy.LM(
        model="openai/gpt-4o",
        api_key=api_key,
        api_base=api_base,
        temperature=0.2,
        max_tokens=256,
    )

    # 3) Configure DSPy global LM (required for GEPA)
    dspy.configure(lm=persona_lm)

    # 4) Build program, judge, dataset
    program = PersonaProgram()
    judge = JudgeProgram()
    dataset = build_dataset()
    trainset = [dspy.Example(**example.__dict__).with_inputs("history", "question") for example in dataset]
    valset = trainset

    # 5) Build GEPA + metric and run a tiny compile
    metric = build_metric(judge, judge_lm=judge_lm)
    gepa = dspy.GEPA(
        metric=metric,
        reflection_lm=reflection_lm,
        teacher_lm=reflection_lm,
        meta_lm=reflection_lm,
        num_threads=2,
        max_metric_calls=8,
    )

    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    # 6) Quick inference
    with dspy.context(lm=persona_lm):
        pred = optimized(history=trainset[0].history, question=trainset[0].question)
    print("Prediction:", getattr(pred, "answer", pred))


if __name__ == "__main__":
    main()
