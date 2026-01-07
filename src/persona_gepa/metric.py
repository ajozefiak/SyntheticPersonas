from __future__ import annotations

from typing import Dict, Optional

from persona_gepa.judge import Judgment, parse_judge_output


def weighted_score(judgment: Judgment, weights: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        if key == "accuracy":
            total += weight * judgment.accuracy
        elif key == "faithfulness":
            total += weight * judgment.faithfulness
        elif key == "tone":
            total += weight * judgment.tone
        elif key == "style":
            total += weight * judgment.style
    return total


def build_metric(judge_module, weights: Dict[str, float]):
    normalized = _normalize_weights(weights)

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        history = getattr(gold, "history", "")
        question = getattr(gold, "question", "")
        reference_answer = getattr(gold, "answer", "")

        candidate_answer = getattr(pred, "answer", None)
        if candidate_answer is None:
            candidate_answer = str(pred)

        judge_pred = judge_module(
            history=history,
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        raw_judgment = getattr(judge_pred, "judgment", judge_pred)
        judgment = parse_judge_output(raw_judgment)
        score = weighted_score(judgment, normalized)
        return {
            "score": score,
            "feedback": judgment.feedback,
            "aspect_scores": judgment.as_dict(),
        }

    return metric


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if total <= 0:
        return {key: 0.0 for key in weights}
    return {key: value / total for key, value in weights.items()}


def evaluate_predictions(predictions: Dict[str, object]) -> Optional[Dict[str, float]]:
    if not predictions:
        return None
    return {
        "mean_score": float(sum(predictions.values())) / max(len(predictions), 1)
    }
