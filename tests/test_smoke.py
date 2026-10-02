"""Offline tests for the HydraDB Hermes memory provider (no Hermes/network)."""

import time

import responses

from hydradb_hermes import HydraDBMemoryProvider
from hydradb_hermes.client import HydraDBClient

BASE = "https://api.hydradb.com"


def _provider():
    client = HydraDBClient(api_key="k", tenant_id="test_beam", sub_tenant_id="")
    return HydraDBMemoryProvider(client=client)


@responses.activate
def test_prefetch_prefers_server_llm_prompt():
    responses.add(
        responses.POST,
        f"{BASE}/query",
        json={
            "success": True,
            "data": {"llm_prompt": "# Query results\n- mascot is Hydrabyte", "chunks": []},
            "error": None,
        },
        status=200,
    )
    out = _provider().prefetch("mascot", session_id="s1")
    assert "mascot is Hydrabyte" in out


@responses.activate
def test_prefetch_falls_back_to_chunks():
    responses.add(
        responses.POST,
        f"{BASE}/query",
        json={
            "success": True,
            "data": {"chunks": [{"chunk_content": "deploy on Fridays", "chunk_uuid": "c1"}]},
            "error": None,
        },
        status=200,
    )
    out = _provider().prefetch("deploy", session_id="s1")
    assert "deploy on Fridays" in out


@responses.activate
def test_sync_turn_ingests_non_blocking():
    responses.add(
        responses.POST,
        f"{BASE}/context/ingest",
        json={"success": True, "data": {"success_count": 1}, "error": None},
        status=200,
    )
    _provider().sync_turn("what is X?", "X is Y", session_id="s1")
    for _ in range(50):
        if responses.calls:
            break
        time.sleep(0.02)
    assert responses.calls
    body = responses.calls[0].request.body
    assert b"app_knowledge" in body


def test_is_available_requires_key_and_tenant(monkeypatch):
    monkeypatch.delenv("HYDRADB_API_KEY", raising=False)
    monkeypatch.delenv("HYDRADB_TENANT_ID", raising=False)
    assert _provider().is_available() is False
    monkeypatch.setenv("HYDRADB_API_KEY", "x")
    assert _provider().is_available() is False
    monkeypatch.setenv("HYDRADB_TENANT_ID", "t")
    assert _provider().is_available() is True


def test_name_and_schema():
    p = _provider()
    assert p.name == "hydradb"
    assert any(f["key"] == "api_key" for f in p.get_config_schema())
