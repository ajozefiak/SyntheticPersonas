from __future__ import annotations

import inspect

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


def build_lm(model: str, temperature: float, max_tokens: int):
    """Build a DSPy LM, falling back across available providers."""
    if hasattr(dspy, "LM"):
        return dspy.LM(model=model, temperature=temperature, max_tokens=max_tokens)

    if hasattr(dspy, "OpenAI"):
        short_model = model.split("/", 1)[-1]
        return dspy.OpenAI(model=short_model, temperature=temperature, max_tokens=max_tokens)

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
