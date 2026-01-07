"""GEPA + DSPy helpers for optimizing synthetic persona prompts."""

from persona_gepa.config import PersonaGEPAConfig
from persona_gepa.data import (
    build_examples,
    build_train_val_examples,
    format_history,
    load_interviews,
    split_interviews,
)
from persona_gepa.infer import run_inference
from persona_gepa.optimize import run_optimization
from persona_gepa.program import PersonaAnswerProgram

__all__ = [
    "PersonaGEPAConfig",
    "PersonaAnswerProgram",
    "build_examples",
    "build_train_val_examples",
    "format_history",
    "load_interviews",
    "split_interviews",
    "run_optimization",
    "run_inference",
]
