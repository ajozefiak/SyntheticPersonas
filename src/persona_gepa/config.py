from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class PersonaGEPAConfig:
    persona_model: str = "openai/gpt-4o"
    judge_model: str = "openai/gpt-4o"
    reflection_model: str = "openai/gpt-4o"

    persona_temperature: float = 0.2
    judge_temperature: float = 0.0
    reflection_temperature: float = 0.2

    persona_max_tokens: int = 512
    judge_max_tokens: int = 512
    reflection_max_tokens: int = 512

    num_threads: int = 8

    cache_dir: str = ".cache/dspy"
    output_dir: str = "artifacts/persona_gepa"
    log_dir: str = "logs/persona_gepa"

    budget: str = "light"
    max_metric_calls: Optional[int] = None

    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy": 0.4,
            "faithfulness": 0.3,
            "tone": 0.15,
            "style": 0.15,
        }
    )

    def resolved_budget(self) -> Dict[str, object]:
        """Return GEPA budget arguments based on config settings."""
        if self.max_metric_calls is not None:
            return {"max_metric_calls": self.max_metric_calls}
        if self.budget:
            return {"auto": self.budget}
        return {"auto": "light"}

    def normalized_weights(self) -> Dict[str, float]:
        total = sum(self.score_weights.values())
        if total <= 0:
            return {key: 0.0 for key in self.score_weights}
        return {key: value / total for key, value in self.score_weights.items()}
