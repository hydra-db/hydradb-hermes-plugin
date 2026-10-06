"""HydraDB as a Hermes Agent memory provider.

Implements the ``MemoryProvider`` contract: ``prefetch`` recalls relevant context
before each turn (auto-recall), ``sync_turn`` persists completed turns
(non-blocking), over the public HydraDB API via the shared client.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional

from .client import HydraDBClient

#: Canonical HydraDB env var -> accepted deprecated aliases. The canonical name
#: wins; the alias keeps older setups (and the pre-rename catalog entry) working.
_ENV_ALIASES = {
    "HYDRADB_DATABASE": ("HYDRADB_TENANT_ID",),
    "HYDRADB_COLLECTION": ("HYDRADB_SUB_TENANT_ID",),
}


def _env(canonical: str, default: str = "") -> str:
    """Resolve a HydraDB env var by canonical name, falling back to aliases."""
    val = os.environ.get(canonical)
    if val is not None:
        return val
    for alias in _ENV_ALIASES.get(canonical, ()):  # pragma: no branch
        val = os.environ.get(alias)
        if val is not None:
            return val
    return default


try:
    from agent.memory_provider import MemoryProvider as _BaseMemoryProvider
except Exception:  # pragma: no cover

    class _BaseMemoryProvider:
        pass


class HydraDBMemoryProvider(_BaseMemoryProvider):
    """Recall/retain backed by HydraDB (public API only)."""

    def __init__(self, client: Optional[HydraDBClient] = None) -> None:
        self._client = client
        self._search_mode = "knowledge"

    @property
    def name(self) -> str:
        return "hydradb"

    def is_available(self) -> bool:
        return bool(os.environ.get("HYDRADB_API_KEY") and _env("HYDRADB_DATABASE"))

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        if self._client is None:
            self._client = HydraDBClient(
                api_key=os.environ.get("HYDRADB_API_KEY", ""),
                tenant_id=_env("HYDRADB_DATABASE"),
                sub_tenant_id=_env("HYDRADB_COLLECTION"),
                base_url=os.environ.get("HYDRADB_BASE_URL", "https://api.hydradb.com"),
            )

    def get_config_schema(self):
        # Hermes reads the `env_var` key to persist each value into .env during
        # `hermes memory setup`; the provider then resolves them from the
        # environment on start (see is_available/initialize). Older
        # HYDRADB_TENANT_ID / HYDRADB_SUB_TENANT_ID remain accepted as aliases.
        return [
            {"key": "api_key", "env_var": "HYDRADB_API_KEY", "secret": True, "required": True},
            {"key": "database", "env_var": "HYDRADB_DATABASE", "required": True},
            {"key": "collection", "env_var": "HYDRADB_COLLECTION", "required": False},
        ]

    def get_tool_schemas(self):
        return []

    def system_prompt_block(self) -> str:
        return "HydraDB provides persistent long-term memory. Relevant memories are recalled automatically before each turn."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not query or self._client is None:
            return ""
        try:
            return self._client.recall_prompt(query, kind=self._search_mode)
        except Exception:
            return ""

    def sync_turn(self, user: str, assistant: str, *, session_id: str = "", messages: Any = None) -> None:
        """Persist a completed turn. MUST be non-blocking, so ingest off-thread."""
        if self._client is None:
            return
        client = self._client

        def _persist() -> None:
            try:
                client.add_conversation(user, assistant, infer=False)
            except Exception:
                pass

        threading.Thread(target=_persist, daemon=True).start()


def register(ctx) -> None:
    ctx.register_memory_provider(HydraDBMemoryProvider())
