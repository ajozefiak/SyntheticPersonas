from __future__ import annotations

import os
from typing import Dict

import inspect

import dspy


class OrgCacheClient:
    """Optional adapter for org UUID cache system.

    TODO: Implement submit and fetch to integrate with the org cache service.
    """

    def submit(self, request: Dict[str, object]) -> str:
        raise NotImplementedError("OrgCacheClient.submit is not implemented.")

    def fetch(self, uuid: str) -> str:
        raise NotImplementedError("OrgCacheClient.fetch is not implemented.")


def configure_dspy_cache(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    configure = getattr(dspy, "configure_cache", None)
    if callable(configure):
        try:
            signature = inspect.signature(configure)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            params = signature.parameters
            if "cache_dir" in params:
                configure(cache_dir=cache_dir)
                return
            if "cache_path" in params:
                configure(cache_path=cache_dir)
                return
            if params:
                configure(cache_dir)
                return

        configure(cache_dir)
