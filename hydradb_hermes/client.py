"""Minimal HydraDB HTTP client - public API only.

Wraps the three public endpoints these integrations need:

* ``POST /query``          - hybrid recall over memory or knowledge (JSON body)
* ``POST /context/ingest`` - write knowledge (multipart ``app_knowledge``)
* ``DELETE /context``      - remove context by id (JSON body)

Ingestion uses the public knowledge path (``app_knowledge``); the memory-family
write route is not exposed on the public API, so text/conversations are stored
as knowledge. Auth is ``Authorization: Bearer <api_key>``. Responses are the
standard ``{"success", "data", "error", "meta"}`` envelope; recall results live
in ``data.chunks``. Chunks are normalised so callers do not depend on the wire
shape.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests

DEFAULT_BASE_URL = "https://api.hydradb.com"


class HydraDBError(RuntimeError):
    """Raised when the HydraDB API returns an error envelope or bad status."""


@dataclass
class HydraChunk:
    """A single recalled chunk, normalised across memory/knowledge results."""

    text: str
    score: float | None = None
    source_title: str = ""
    id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class HydraDBClient:
    """Thin, dependency-light client for HydraDB recall and ingestion."""

    def __init__(
        self,
        api_key: str,
        tenant_id: str,
        sub_tenant_id: str = "",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        if not tenant_id:
            raise ValueError("tenant_id is required")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.sub_tenant_id = sub_tenant_id
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _unwrap(self, resp: requests.Response) -> dict[str, Any]:
        try:
            body = resp.json()
        except ValueError:
            raise HydraDBError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if not body.get("success", False):
            err = body.get("error") or {}
            raise HydraDBError(
                f"{err.get('code', 'ERROR')}: {err.get('message', resp.text[:200])}"
            )
        return body.get("data") or {}

    def query(
        self,
        query: str,
        kind: str = "memory",
        mode: str = "fast",
        max_results: int = 6,
        alpha: float = 0.8,
        recency_bias: float = 0.0,
        graph_context: bool = True,
    ) -> list[HydraChunk]:
        """Hybrid recall. ``kind`` is ``"memory"`` or ``"knowledge"``."""
        payload = {
            "tenant_id": self.tenant_id,
            "sub_tenant_id": self.sub_tenant_id,
            "query": query,
            "kind": kind,
            "mode": mode,
            "max_results": max_results,
            "alpha": alpha,
            "recency_bias": recency_bias,
            "graph_context": graph_context,
        }
        resp = self._session.post(
            f"{self.base_url}/query",
            headers={**self._auth_headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        data = self._unwrap(resp)
        return [_to_chunk(c) for c in (data.get("chunks") or [])]

    def recall_prompt(
        self,
        query: str,
        kind: str = "knowledge",
        mode: str = "fast",
        max_results: int = 6,
    ) -> str:
        """Hybrid recall as a ready-to-inject string.

        Prefers the server-rendered ``llm_prompt`` when present; otherwise builds
        a compact block from the recalled chunks. Returns ``""`` when nothing is
        recalled.
        """
        payload = {
            "tenant_id": self.tenant_id,
            "sub_tenant_id": self.sub_tenant_id,
            "query": query,
            "kind": kind,
            "mode": mode,
            "max_results": max_results,
        }
        resp = self._session.post(
            f"{self.base_url}/query",
            headers={**self._auth_headers, "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        data = self._unwrap(resp)
        prompt = data.get("llm_prompt")
        if isinstance(prompt, str) and prompt.strip():
            return prompt
        chunks = [_to_chunk(c) for c in (data.get("chunks") or [])]
        lines = [f"- {c.text}" for c in chunks if c.text]
        return "Relevant HydraDB memory:\n" + "\n".join(lines) if lines else ""

    def add_knowledge(self, items: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Ingest ``app_knowledge`` items via the public ``/context/ingest``.

        Each item is ``{"id", "title", "content": {"text": ...}, "metadata",
        "infer"}``. Ingestion is asynchronous: a success here means the sources
        were accepted/queued, not yet indexed.
        """
        fields = {
            "tenant_id": (None, self.tenant_id),
            "sub_tenant_id": (None, self.sub_tenant_id),
            "kind": (None, "knowledge"),
            "app_knowledge": (None, json.dumps(list(items))),
        }
        resp = self._session.post(
            f"{self.base_url}/context/ingest",
            headers=self._auth_headers,
            files=fields,
            timeout=self.timeout,
        )
        return self._unwrap(resp)

    def add_text(
        self,
        text: str,
        infer: bool = True,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        """Store a single note as knowledge. Returns the ingest ack."""
        item = {
            "id": source_id or f"note-{uuid.uuid4().hex}",
            "title": title or "note",
            "content": {"text": text},
            "metadata": metadata or {},
            "infer": infer,
        }
        return self.add_knowledge([item])

    def add_conversation(
        self, user: str, assistant: str, infer: bool = True
    ) -> dict[str, Any]:
        """Store a user/assistant turn as knowledge."""
        return self.add_text(
            f"User: {user}\nAssistant: {assistant}", infer=infer, title="conversation"
        )

    def delete(self, ids: list[str], kind: str = "knowledge") -> dict[str, Any]:
        ids = [i for i in (ids or []) if i]
        if not ids:
            return {"deleted_count": 0, "results": []}
        resp = self._session.delete(
            f"{self.base_url}/context",
            headers={**self._auth_headers, "Content-Type": "application/json"},
            json={
                "tenant_id": self.tenant_id,
                "sub_tenant_id": self.sub_tenant_id,
                "ids": ids,
                "kind": kind,
            },
            timeout=self.timeout,
        )
        return self._unwrap(resp)


def _to_chunk(raw: dict[str, Any]) -> HydraChunk:
    """Normalise a wire chunk into :class:`HydraChunk`.

    ``chunk_content`` is often a stored envelope whose prose lives at
    ``content.text``; recall must yield the prose, not the JSON scaffolding
    (mirrors the plugin's normalizeRetrievalResponse). Content that merely looks
    JSON-ish but does not parse is passed through untouched.
    """
    raw_text = raw.get("chunk_content")
    if raw_text is None:
        raw_text = raw.get("text") or raw.get("content") or ""
    return HydraChunk(
        text=_extract_text(raw_text),
        score=raw.get("relevancy_score") or raw.get("score") or raw.get("relevance_score"),
        source_title=raw.get("source_title") or raw.get("title") or "",
        id=raw.get("chunk_uuid") or raw.get("id") or "",
        metadata=raw.get("metadata") or {},
    )


def _extract_text(value: Any) -> str:
    """Unwrap a HydraDB content envelope to its prose, else return as-is."""
    if isinstance(value, dict):
        return _prose_from_envelope(value) or json.dumps(value)
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
        except ValueError:
            return value
        prose = _prose_from_envelope(obj)
        if prose is not None:
            return prose
    return value


def _prose_from_envelope(obj: Any) -> str | None:
    if not isinstance(obj, dict):
        return None
    content = obj.get("content")
    if isinstance(content, dict):
        for key in ("text", "markdown"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                return val
    if isinstance(content, str) and content.strip():
        return content
    top = obj.get("text")
    if isinstance(top, str) and top.strip():
        return top
    return None
