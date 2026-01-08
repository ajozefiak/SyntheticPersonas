from __future__ import annotations

import inspect
import os

import dspy


def _get_configured_lm():
    settings = getattr(dspy, "settings", None)
    if settings is None:
        return None
    if hasattr(settings, "lm"):
        return getattr(settings, "lm")
    if hasattr(settings, "get"):
        try:
            return settings.get("lm")
        except TypeError:
            return None
    return None


def ensure_dspy_lm_configured() -> None:
    configured = _get_configured_lm()
    if configured is None:
        raise RuntimeError(
            "No LM is loaded. Please configure the LM using "
            "dspy.configure(lm=dspy.LM(...))."
        )


def configure_dspy_lm(lm) -> None:
    if lm is None:
        raise RuntimeError(
            "No LM is loaded. Please configure the LM using "
            "dspy.configure(lm=dspy.LM(...))."
        )
    configure = getattr(dspy, "configure", None)
    if not callable(configure):
        raise RuntimeError("DSPy configure() is unavailable; cannot set LM.")
    configure(lm=lm)
    ensure_dspy_lm_configured()


def build_lm(
    model: str,
    temperature: float,
    max_tokens: int,
    api_base: str | None = None,
    api_key: str | None = None,
):
    """Build a DSPy LM, falling back across available providers."""
    api_base = api_base or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    lm_kwargs = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_base:
        lm_kwargs["api_base"] = api_base
    if api_key:
        lm_kwargs["api_key"] = api_key

    if hasattr(dspy, "LM"):
        return dspy.LM(**filter_kwargs(dspy.LM, lm_kwargs))

    if hasattr(dspy, "OpenAI"):
        short_model = model.split("/", 1)[-1]
        lm_kwargs["model"] = short_model
        return dspy.OpenAI(**filter_kwargs(dspy.OpenAI, lm_kwargs))

    raise RuntimeError("No compatible DSPy LM backend found.")


def filter_kwargs(callable_obj, kwargs):
    """Filter kwargs to those accepted by callable_obj."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}
