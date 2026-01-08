from types import SimpleNamespace

import pytest

dspy = pytest.importorskip("dspy")

from persona_gepa.metric import build_metric


class DummyJudge:
    def __call__(self, **kwargs):
        return SimpleNamespace(
            judgment='{"accuracy": 1, "faithfulness": 0.5, "tone": 0.0, "style": 0.0, "feedback": "ok"}'
        )


def test_metric_shape_and_signature():
    metric = build_metric(DummyJudge(), {"accuracy": 1.0})
    gold = SimpleNamespace(history="", question="Q", answer="A")
    pred = SimpleNamespace(answer="A")

    result = metric(gold, pred, trace=None, pred_name=None, pred_trace=None)

    assert hasattr(result, "score")
    assert hasattr(result, "feedback")
    assert isinstance(result.score, float)
    assert result.feedback == "ok"
