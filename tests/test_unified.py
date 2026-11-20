import json

import pytest
import responses

from hydradb_hermes.client import HydraDBClient, HydraDBError

BASE = "https://api.hydradb.com"


def client(**kwargs):
    return HydraDBClient(api_key="offline-key", tenant_id="db", sub_tenant_id="col", **kwargs)


def respond(data, meta=None, status=200, method=responses.POST, path="/query"):
    responses.add(method, BASE + path, json={"success": True, "data": data, "meta": meta or {}}, status=status)


@pytest.mark.parametrize("layout", ["auto", "UNIFIED", "", None])
def test_layout_requires_explicit_supported_value(layout):
    with pytest.raises(ValueError, match="layout"):
        client(layout=layout)


@responses.activate
def test_unified_query_retains_identity_content_and_root_data_by_shape():
    text = '  {"content": {"text": "do not unwrap"}}  '
    data = {
        "chunks": [{
            "chunk_id": "chunk-1", "context_id": "context-1", "content": text,
            "score": 0.0, "enrichment": "separate facts", "received_at": "2026-11-19T00:00:00Z",
        }],
        "graph": [{"triplets": [], "path_summary": "a path"}],
        "forceful_relations": [{
            "via": {"from": "context-1", "to": "linked"},
            "chunk": {"chunk_id": "linked-chunk", "context_id": "linked", "content": "linked content", "score": 0.5},
        }],
        "llm_prompt": "  verbatim prompt\n",
        "tabular_data": {"status": "ok", "records": [{"n": 2}]},
    }
    respond(data, {"source_type": "unified", "warnings": [{"code": "notice"}]})
    result = client().query_result("question")
    chunk = result.chunks[0]
    assert chunk.text == text
    assert chunk.id == "chunk-1"
    assert chunk.context_id == "context-1"
    assert chunk.score == 0.0
    assert chunk.enrichment == "separate facts"
    assert chunk.received_at == "2026-11-19T00:00:00Z"
    assert chunk.source_title == ""
    assert chunk.metadata == {}
    assert result.data == data
    assert result.meta == {"source_type": "unified", "warnings": [{"code": "notice"}]}
    assert result.tabular_data == data["tabular_data"]
    assert result.llm_prompt == data["llm_prompt"]


@responses.activate
def test_unified_query_uses_canonical_scope_without_split_selector():
    respond({"chunks": [{"chunk_uuid": "old-chunk", "chunk_content": "split response"}]})
    chunks = client(layout="unified").query("question", kind="memory")
    assert chunks[0].id == "old-chunk"
    assert chunks[0].text == "split response"
    body = json.loads(responses.calls[0].request.body)
    assert body == {
        "database": "db", "collection": "col", "query": "question",
        "mode": "fast", "max_results": 6, "alpha": 0.8, "recency_bias": 0.0, "graph_context": True,
    }


@responses.activate
def test_split_structured_result_retains_tabular_and_envelope_metadata():
    data = {"chunks": [], "tabular_data": {"status": "ok", "records": [{"count": 1}]}}
    respond(data, {"request_id": "offline"})
    result = client().query_result("table")
    assert result.chunks == []
    assert result.data == data
    assert result.tabular_data == data["tabular_data"]
    assert result.meta == {"request_id": "offline"}
    body = json.loads(responses.calls[0].request.body)
    assert body["type"] == "memory"
    assert body["tenant_id"] == "db"


@responses.activate
def test_unified_text_translates_metadata_and_preserves_explicit_id():
    respond({"success_count": 1}, path="/context/ingest", status=202)
    assert client(layout="unified").add_text(
        "note", infer=False, title="title", metadata={"team": "eng"}, source_id="stable"
    ) == {"success_count": 1}
    request = responses.calls[0].request
    assert request.headers["Content-Type"] == "application/json"
    body = json.loads(request.body)
    assert body == {
        "database": "db", "collection": "col",
        "context": [{"context_id": "stable", "title": "title", "text": "note",
                     "attributes": {"team": "eng"}, "enrich": False}],
    }


@responses.activate
def test_unified_raw_items_translate_both_metadata_names_without_mutation():
    respond({"success_count": 1}, path="/context/ingest", status=202)
    item = {"id": "stable", "title": "title", "content": {"text": "body"},
            "metadata": {"team": "eng"}, "additional_metadata": {"author": "a"}, "infer": False}
    original = json.loads(json.dumps(item))
    client(layout="unified").add_knowledge(iter([item]))
    assert item == original
    assert json.loads(responses.calls[0].request.body)["context"] == [{
        "context_id": "stable", "title": "title", "text": "body",
        "attributes": {"team": "eng"}, "custom_attributes": {"author": "a"}, "enrich": False,
    }]


@pytest.mark.parametrize("item", [
    {"content": {"text": "body", "html_base64": "unsupported"}},
    {"content": {"text": "body"}, "unknown": "must not disappear"},
    {"content": {"text": "body"}, "context_category": "removed"},
])
@responses.activate
def test_unified_raw_items_reject_unsupported_fields_without_network(item):
    with pytest.raises(ValueError):
        client(layout="unified").add_knowledge([item])
    assert not responses.calls


@responses.activate
def test_unified_conversations_use_roles_and_distinct_context_ids():
    for _ in range(2):
        respond({"success_count": 1}, path="/context/ingest", status=202)
    c = client(layout="unified")
    for _ in range(2):
        c.add_conversation("hello", "answer", infer=False)
    items = [json.loads(call.request.body)["context"][0] for call in responses.calls]
    for item in items:
        assert item["conversation"] == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "answer"}]
        assert "text" not in item
        assert item["enrich"] is False
        assert item["context_id"]
    assert items[0]["context_id"] != items[1]["context_id"]


@responses.activate
def test_unified_delete_uses_context_ids_and_no_split_selector():
    respond({"deleted_count": 1}, method=responses.DELETE, path="/context")
    assert client(layout="unified").delete(["context-1", ""], kind="memory") == {"deleted_count": 1}
    assert json.loads(responses.calls[0].request.body) == {
        "database": "db", "collection": "col", "ids": ["context-1"],
    }


@pytest.mark.parametrize("layout", ["split", "unified"])
@responses.activate
def test_failed_write_never_retries_another_layout(layout):
    responses.add(responses.POST, BASE + "/context/ingest",
                  json={"success": False, "error": {"code": "INVALID", "message": "no"}},
                  status=400)
    with pytest.raises(HydraDBError):
        client(layout=layout).add_text("note")
    assert len(responses.calls) == 1
    request = responses.calls[0].request
    assert ("application/json" in request.headers["Content-Type"]) == (layout == "unified")


@pytest.mark.parametrize("user, assistant, expected", [
    ("", "  answer\n", [{"role": "assistant", "content": "  answer\n"}]),
    (" \t\n", "  answer\n", [{"role": "assistant", "content": "  answer\n"}]),
    ("  question\n", "", [{"role": "user", "content": "  question\n"}]),
    ("  question\n", " \t\n", [{"role": "user", "content": "  question\n"}]),
    ("  question\n", "  answer\n", [
        {"role": "user", "content": "  question\n"},
        {"role": "assistant", "content": "  answer\n"},
    ]),
])
@responses.activate
def test_unified_conversation_omits_only_blank_counterparts(user, assistant, expected):
    respond({"success_count": 1}, path="/context/ingest", status=202)
    client(layout="unified").add_conversation(user, assistant, infer=False)
    assert len(responses.calls) == 1
    body = json.loads(responses.calls[0].request.body)
    assert body["database"] == "db"
    assert body["collection"] == "col"
    assert "type" not in body and "kind" not in body
    item = body["context"][0]
    assert item["conversation"] == expected
    assert item["enrich"] is False
    assert item["title"] == "conversation"
    assert item["context_id"].startswith("note-")
    assert "text" not in item


@pytest.mark.parametrize("user, assistant", [
    ("", ""), (" \t", "\n"), ("", " \n"), (" \t", ""),
])
@responses.activate
def test_unified_conversation_rejects_both_blank_before_http(user, assistant):
    with pytest.raises(ValueError, match="conversation"):
        client(layout="unified").add_conversation(user, assistant)
    assert not responses.calls


@pytest.mark.parametrize("layout", [None, "split"])
@pytest.mark.parametrize("user, assistant", [
    ("", "  answer\n"), ("  question\n", ""), (" \t", "\n"),
])
@responses.activate
def test_split_conversation_keeps_exact_transcript_for_blank_counterparts(layout, user, assistant):
    from email import message_from_bytes

    respond({"success_count": 1}, path="/context/ingest", status=202)
    c = client() if layout is None else client(layout=layout)
    c.add_conversation(user, assistant, infer=False)
    assert len(responses.calls) == 1
    request = responses.calls[0].request
    message = message_from_bytes(
        b"Content-Type: " + request.headers["Content-Type"].encode() + b"\r\n\r\n" + request.body
    )
    fields = {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
        for part in message.walk() if part.get_content_disposition() == "form-data"
    }
    item = json.loads(fields["app_knowledge"])[0]
    assert item["content"]["text"] == f"User: {user}\nAssistant: {assistant}"
    assert item["infer"] is False
    assert fields["type"] == b"knowledge"
