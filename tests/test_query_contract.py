"""Offline query request and response contract checks."""

import json

import responses

from hydradb_hermes.client import HydraDBClient


@responses.activate
def test_structured_query_preserves_response_and_options():
    data = {
        "chunks": [{"chunk_content": "hello", "metadata": {"old": 1},
                    "chunk_metadata": {"page": 2}}],
        "tabular_data": {"status": "ok", "records": [{"value": 1}], "sql": "select 1"},
        "temporal_facts": [{"observed_at": 1767225600}],
        "graph": {"entities": [{"database": "db", "collection": "col",
                                  "source_id": "s", "read_status": "complete"}]},
    }
    meta = {"warnings": [{"code": "PARTIAL", "message": "partial",
                           "collections": ["col"]}]}
    responses.add(responses.POST, "https://api.hydradb.com/query",
                  json={"success": True, "data": data, "meta": meta})
    client = HydraDBClient("key", "tenant")
    result = client.query_result("question", include_table_sql=False, table_limit=0)
    payload = json.loads(responses.calls[0].request.body)
    assert payload["include_table_sql"] is False
    assert payload["table_limit"] == 0
    assert result.chunks[0].metadata == {"old": 1}
    assert result.chunks[0].chunk_metadata == {"page": 2}
    assert result.data == data
    assert result.meta == meta
    assert result.tabular_data == data["tabular_data"]
    assert result.warnings == meta["warnings"]
    assert result.data["temporal_facts"][0]["observed_at"] == 1767225600
    assert result.data["graph"]["entities"][0]["read_status"] == "complete"


@responses.activate
def test_existing_query_keeps_list_contract_and_omits_unspecified_options():
    responses.add(responses.POST, "https://api.hydradb.com/query",
                  json={"success": True, "data": {"chunks": [{"text": "hello"}]}})
    client = HydraDBClient("key", "tenant")
    chunks = client.query("question")
    payload = json.loads(responses.calls[0].request.body)
    assert "include_table_sql" not in payload and "table_limit" not in payload
    assert isinstance(chunks, list) and chunks[0].text == "hello"


@responses.activate
def test_list_query_accepts_table_options_and_chunk_metadata():
    responses.add(responses.POST, "https://api.hydradb.com/query",
                  json={"success": True, "data": {"chunks": [{"text": "hello", "chunk_metadata": {"x": 1}}]}})
    client = HydraDBClient("key", "tenant")
    chunks = client.query("question", include_table_sql=True, table_limit=3)
    payload = json.loads(responses.calls[0].request.body)
    assert payload["include_table_sql"] is True and payload["table_limit"] == 3
    assert chunks[0].chunk_metadata == {"x": 1}
