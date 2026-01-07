from __future__ import annotations

import argparse
import importlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Tuple

import dspy

from persona_gepa.artifacts import save_artifact
from persona_gepa.cache import configure_dspy_cache
from persona_gepa.config import PersonaGEPAConfig
from persona_gepa.data import build_examples, load_interviews, split_interviews
from persona_gepa.judge import JudgeProgram, parse_judge_output
from persona_gepa.metric import build_metric, weighted_score
from persona_gepa.program import PersonaAnswerProgram
from persona_gepa.utils import build_lm, filter_kwargs


def _configure_lm(persona_lm):
    settings = getattr(dspy, "settings", None)
    if settings is not None and hasattr(settings, "configure"):
        settings.configure(lm=persona_lm)


def _load_interviews_with_hook(path: str, loader_path: str | None):
    if not loader_path:
        return load_interviews(path)
    module_name, func_name = loader_path.split(":", 1)
    module = importlib.import_module(module_name)
    loader: Callable[[str], List[List[dict]]] = getattr(module, func_name)
    return loader(path)


def _evaluate_program(
    program: PersonaAnswerProgram,
    valset: Iterable,
    judge: JudgeProgram,
    weights: Dict[str, float],
    num_threads: int,
) -> Dict[str, float]:
    valset = list(valset)
    if not valset:
        return {}

    normalized = weights
    scores: List[float] = []
    aspect_totals = {"accuracy": 0.0, "faithfulness": 0.0, "tone": 0.0, "style": 0.0}

    def _score_example(example):
        pred = program(
            history=getattr(example, "history", ""),
            question=getattr(example, "question", ""),
            persona_profile=getattr(example, "persona_profile", ""),
        )
        candidate_answer = getattr(pred, "answer", str(pred))
        judge_pred = judge(
            history=getattr(example, "history", ""),
            question=getattr(example, "question", ""),
            reference_answer=getattr(example, "answer", ""),
            candidate_answer=candidate_answer,
        )
        raw_judgment = getattr(judge_pred, "judgment", judge_pred)
        judgment = parse_judge_output(raw_judgment)
        score = weighted_score(judgment, normalized)
        return score, judgment

    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as executor:
        for score, judgment in executor.map(_score_example, valset):
            scores.append(score)
            aspect_totals["accuracy"] += judgment.accuracy
            aspect_totals["faithfulness"] += judgment.faithfulness
            aspect_totals["tone"] += judgment.tone
            aspect_totals["style"] += judgment.style

    count = max(len(scores), 1)
    return {
        "mean_score": sum(scores) / count,
        "mean_accuracy": aspect_totals["accuracy"] / count,
        "mean_faithfulness": aspect_totals["faithfulness"] / count,
        "mean_tone": aspect_totals["tone"] / count,
        "mean_style": aspect_totals["style"] / count,
        "count": float(len(scores)),
    }


def run_optimization(
    config: PersonaGEPAConfig,
    trainset: List,
    valset: List,
) -> Tuple[PersonaAnswerProgram, str, Dict[str, float]]:
    os.makedirs(config.output_dir, exist_ok=True)
    configure_dspy_cache(config.cache_dir)

    persona_lm = build_lm(
        config.persona_model, config.persona_temperature, config.persona_max_tokens
    )
    judge_lm = build_lm(
        config.judge_model, config.judge_temperature, config.judge_max_tokens
    )
    reflection_lm = build_lm(
        config.reflection_model,
        config.reflection_temperature,
        config.reflection_max_tokens,
    )

    _configure_lm(persona_lm)

    program = PersonaAnswerProgram(lm=persona_lm)
    judge = JudgeProgram(lm=judge_lm)

    metric = build_metric(judge, config.normalized_weights())

    gepa_kwargs = {
        "num_threads": config.num_threads,
        "metric": metric,
        "reflection_lm": reflection_lm,
        "teacher_lm": reflection_lm,
        "meta_lm": reflection_lm,
    }
    gepa = dspy.GEPA(**filter_kwargs(dspy.GEPA, gepa_kwargs))

    compile_kwargs = {
        **config.resolved_budget(),
        "metric": metric,
    }
    compile_kwargs = filter_kwargs(gepa.compile, compile_kwargs)

    optimized_program = gepa.compile(
        program, trainset=trainset, valset=valset, **compile_kwargs
    )

    artifact_path = os.path.join(config.output_dir, "persona_gepa_artifact.json")
    metadata = {
        "persona_model": config.persona_model,
        "judge_model": config.judge_model,
        "reflection_model": config.reflection_model,
        "budget": config.budget,
        "max_metric_calls": config.max_metric_calls,
    }
    save_artifact(optimized_program, artifact_path, metadata=metadata)

    report = _evaluate_program(
        optimized_program,
        valset,
        judge,
        config.normalized_weights(),
        config.num_threads,
    )
    if report:
        report_path = os.path.join(config.output_dir, "validation_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    return optimized_program, artifact_path, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GEPA optimization for personas.")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-path", help="JSON/JSONL path to split into train/val.")
    data_group.add_argument("--train-path", help="JSON/JSONL path for training interviews.")
    parser.add_argument("--val-path", help="JSON/JSONL path for validation interviews.")
    parser.add_argument("--loader", help="Optional loader hook module:function.")

    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--persona-model", default="openai/gpt-4o")
    parser.add_argument("--judge-model", default="openai/gpt-4o")
    parser.add_argument("--reflection-model", default="openai/gpt-4o")

    parser.add_argument("--persona-temperature", type=float, default=0.2)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--reflection-temperature", type=float, default=0.2)

    parser.add_argument("--persona-max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=512)
    parser.add_argument("--reflection-max-tokens", type=int, default=512)

    parser.add_argument("--budget", default="light", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-metric-calls", type=int)
    parser.add_argument("--num-threads", type=int, default=8)

    parser.add_argument("--cache-dir", default=".cache/dspy")
    parser.add_argument("--output-dir", default="artifacts/persona_gepa")
    parser.add_argument("--log-dir", default="logs/persona_gepa")

    parser.add_argument("--weight-accuracy", type=float, default=0.4)
    parser.add_argument("--weight-faithfulness", type=float, default=0.3)
    parser.add_argument("--weight-tone", type=float, default=0.15)
    parser.add_argument("--weight-style", type=float, default=0.15)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.train_path:
        train_interviews = _load_interviews_with_hook(args.train_path, args.loader)
        if not args.val_path:
            raise SystemExit("--val-path is required when using --train-path")
        val_interviews = _load_interviews_with_hook(args.val_path, args.loader)
    else:
        interviews = _load_interviews_with_hook(args.data_path, args.loader)
        train_interviews, val_interviews = split_interviews(
            interviews, val_ratio=args.val_ratio, seed=args.seed
        )

    trainset = build_examples(train_interviews)
    valset = build_examples(val_interviews)

    config = PersonaGEPAConfig(
        persona_model=args.persona_model,
        judge_model=args.judge_model,
        reflection_model=args.reflection_model,
        persona_temperature=args.persona_temperature,
        judge_temperature=args.judge_temperature,
        reflection_temperature=args.reflection_temperature,
        persona_max_tokens=args.persona_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        reflection_max_tokens=args.reflection_max_tokens,
        budget=args.budget,
        max_metric_calls=args.max_metric_calls,
        num_threads=args.num_threads,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        score_weights={
            "accuracy": args.weight_accuracy,
            "faithfulness": args.weight_faithfulness,
            "tone": args.weight_tone,
            "style": args.weight_style,
        },
    )

    _, artifact_path, report = run_optimization(config, trainset, valset)
    summary = {"artifact_path": artifact_path, "validation_report": report}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
