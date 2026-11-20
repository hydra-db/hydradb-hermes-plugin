import json
import threading

import pytest
import responses

from hydradb_hermes import HydraDBMemoryProvider
from hydradb_hermes.client import HydraDBClient

BASE = "https://api.hydradb.com"


@pytest.mark.parametrize("prompt", ["  exact\n prompt  ", " \n", ""])
@responses.activate
def test_prefetch_injects_provided_prompt_verbatim(prompt):
    responses.add(responses.POST, BASE + "/query", json={
        "success": True, "data": {"llm_prompt": prompt, "chunks": [{"content": "not a fallback", "chunk_id": "c", "context_id": "s"}]},
    })
    provider = HydraDBMemoryProvider(HydraDBClient("k", "db", layout="unified"))
    assert provider.prefetch("question") == prompt
    assert "type" not in json.loads(responses.calls[0].request.body)


@pytest.mark.parametrize("prompt", [" \n", ""])
@responses.activate
def test_split_recall_falls_back_to_chunks_on_empty_prompt(prompt):
    responses.add(responses.POST, BASE + "/query", json={
        "success": True, "data": {"llm_prompt": prompt, "chunks": [{"content": "fallback body", "chunk_id": "c", "context_id": "s"}]},
    })
    provider = HydraDBMemoryProvider(HydraDBClient("k", "db", layout="split"))
    assert provider.prefetch("question") == "Relevant HydraDB memory:\n- fallback body"


def test_provider_configures_explicit_layout(monkeypatch):
    monkeypatch.setenv("HYDRADB_API_KEY", "k")
    monkeypatch.setenv("HYDRADB_DATABASE", "db")
    monkeypatch.setenv("HYDRADB_LAYOUT", "unified")
    provider = HydraDBMemoryProvider()
    provider.initialize("session")
    assert provider._client.layout == "unified"
    assert any(field["env_var"] == "HYDRADB_LAYOUT" for field in provider.get_config_schema())


@responses.activate
def test_sync_turn_keeps_nonblocking_host_contract_with_unified_write():
    done = threading.Event()
    captured = []
    def acknowledge(request):
        captured.append(json.loads(request.body))
        done.set()
        return 202, {}, json.dumps({"success": True, "data": {"success_count": 1}})
    responses.add_callback(responses.POST, BASE + "/context/ingest", callback=acknowledge)
    provider = HydraDBMemoryProvider(HydraDBClient("k", "db", layout="unified"))
    assert provider.sync_turn("hello", "answer", session_id="s") is None
    assert done.wait(2)
    item = captured[0]["context"][0]
    assert item["conversation"] == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"}]


@pytest.mark.parametrize("user, assistant, expected", [
    ("", "  answer\n", [{"role": "assistant", "content": "  answer\n"}]),
    ("  question\n", "", [{"role": "user", "content": "  question\n"}]),
])
@responses.activate
def test_sync_turn_persists_one_sided_unified_conversation_off_thread(user, assistant, expected):
    done = threading.Event()
    captured = []
    worker_ids = []
    caller_id = threading.get_ident()

    def acknowledge(request):
        captured.append(json.loads(request.body))
        worker_ids.append(threading.get_ident())
        done.set()
        return 202, {}, json.dumps({"success": True, "data": {"success_count": 1}})

    responses.add_callback(responses.POST, BASE + "/context/ingest", callback=acknowledge)
    provider = HydraDBMemoryProvider(HydraDBClient("k", "db", layout="unified"))
    assert provider.sync_turn(user, assistant, session_id="s") is None
    assert done.wait(2)
    assert worker_ids[0] != caller_id
    item = captured[0]["context"][0]
    assert item["conversation"] == expected
    assert item["enrich"] is False
    assert item["context_id"].startswith("note-")
