from __future__ import annotations

import os
from typing import Dict

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
        configure(cache_dir=cache_dir)
