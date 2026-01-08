# DSPy + GEPA audit (2026-01-08)

## Sources reviewed
- https://dspy.ai/learn/programming/language_models/
- https://dspy.ai/api/modules/configure_cache/
- https://dspy.ai/learn/optimization/GEPA/
- https://github.com/gepa-ai/gepa

## What was off vs official docs
- LM configuration: DSPy expects `dspy.configure(lm=...)` before running programs. We were constructing LMs and attaching them to predictors but never setting the global LM, which can surface as "No LM is loaded" in GEPA threads.
- Cache configuration: DSPy `configure_cache` uses `disk_cache_dir` in current docs; our helper only checked `cache_dir` / `cache_path` and could pass the cache directory as a positional arg, which can misconfigure caching.
- Multi-LM usage: DSPy supports scoped LM overrides via `dspy.context(lm=...)`. We did not scope persona/judge calls, which can fall back to the global LM in threaded contexts.
- GEPA metric feedback: The docs expect a metric that returns a score (float) or a score+feedback `dspy.Prediction`. We were returning extra fields beyond score/feedback.

## Changes applied
- Added `configure_dspy_lm` and a defensive LM check, and called it at the start of optimization and inference.
- Updated cache configuration to prefer `disk_cache_dir` when available.
- Wrapped persona/judge calls in `dspy.context(lm=...)` where a non-default LM is required.
- Updated the GEPA metric return to a `dspy.Prediction(score=..., feedback=...)` to match the documented ScoreWithFeedback format.
- Added tests to verify LM configuration in the optimizer and that program forward passes through predictions.
