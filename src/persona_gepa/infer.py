from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

import dspy

from persona_gepa.artifacts import load_program
from persona_gepa.cache import configure_dspy_cache
from persona_gepa.config import PersonaGEPAConfig
from persona_gepa.utils import build_lm, configure_dspy_lm


def run_inference(
    config: PersonaGEPAConfig,
    artifact_path: str,
    history: str,
    question: str,
    persona_profile: str = "",
) -> str:
    configure_dspy_cache(config.cache_dir)
    persona_lm = build_lm(
        config.persona_model,
        config.persona_temperature,
        config.persona_max_tokens,
        api_base=config.api_base,
    )
    configure_dspy_lm(persona_lm)
    program = load_program(artifact_path, lm=persona_lm)
    context = getattr(dspy, "context", None)
    if callable(context):
        with context(lm=persona_lm):
            prediction = program(
                history=history, question=question, persona_profile=persona_profile
            )
    else:
        prediction = program(
            history=history, question=question, persona_profile=persona_profile
        )
    return getattr(prediction, "answer", str(prediction))


def _load_input(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with optimized persona.")
    parser.add_argument("--artifact-path", required=True)

    parser.add_argument("--history", help="Transcript history string.")
    parser.add_argument("--question", help="Current question.")
    parser.add_argument("--persona-profile", default="")
    parser.add_argument("--input-path", help="JSON file with history/question keys.")

    parser.add_argument("--persona-model", default="openai/gpt-4o")
    parser.add_argument("--persona-temperature", type=float, default=0.2)
    parser.add_argument("--persona-max-tokens", type=int, default=512)

    parser.add_argument(
        "--api-base",
        help="Optional API base URL for OpenAI-compatible endpoints.",
    )

    parser.add_argument("--cache-dir", default=".cache/dspy")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.input_path:
        payload = _load_input(args.input_path)
        history = payload.get("history", "")
        question = payload.get("question", "")
        persona_profile = payload.get("persona_profile", "")
    else:
        history = args.history or ""
        question = args.question or ""
        persona_profile = args.persona_profile

    if not question:
        raise SystemExit("Question is required (use --question or --input-path).")

    config = PersonaGEPAConfig(
        persona_model=args.persona_model,
        persona_temperature=args.persona_temperature,
        persona_max_tokens=args.persona_max_tokens,
        api_base=args.api_base,
        cache_dir=args.cache_dir,
    )

    answer = run_inference(config, args.artifact_path, history, question, persona_profile)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
