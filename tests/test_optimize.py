from types import SimpleNamespace

import pytest

dspy = pytest.importorskip("dspy")

from persona_gepa.config import PersonaGEPAConfig
from persona_gepa import optimize as optimize_module


def test_run_optimization_configures_lm(monkeypatch, tmp_path):
    persona_lm = object()
    judge_lm = object()
    reflection_lm = object()
    lm_iter = iter([persona_lm, judge_lm, reflection_lm])
    configure_calls = {}

    def fake_build_lm(*_args, **_kwargs):
        return next(lm_iter)

    def fake_configure_dspy_lm(lm):
        configure_calls["lm"] = lm

    class DummyGEPA:
        def __init__(self, **_kwargs):
            pass

        def compile(self, program, trainset=None, valset=None, **_kwargs):
            return program

    monkeypatch.setattr(optimize_module, "build_lm", fake_build_lm)
    monkeypatch.setattr(optimize_module, "configure_dspy_lm", fake_configure_dspy_lm)
    monkeypatch.setattr(optimize_module, "build_metric", lambda *_a, **_k: lambda *_: {"score": 0.0, "feedback": ""})
    monkeypatch.setattr(optimize_module, "PersonaAnswerProgram", lambda lm=None: SimpleNamespace())
    monkeypatch.setattr(optimize_module, "JudgeProgram", lambda lm=None: SimpleNamespace())
    monkeypatch.setattr(optimize_module, "save_artifact", lambda *_a, **_k: str(tmp_path / "artifact.json"))
    monkeypatch.setattr(optimize_module, "_evaluate_program", lambda *_a, **_k: {})
    monkeypatch.setattr(optimize_module.dspy, "GEPA", DummyGEPA)

    config = PersonaGEPAConfig(
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
    )

    optimize_module.run_optimization(config, trainset=[], valset=[])

    assert configure_calls["lm"] is persona_lm
