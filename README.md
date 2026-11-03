<p align="center">
  <img src="https://raw.githubusercontent.com/hydra-db/hydradb-hermes-plugin/main/assets/icon.png" width="96" alt="HydraDB" />
</p>

# HydraDB for Hermes Agent

A [Hermes Agent](https://hermes-agent.nousresearch.com/docs/) **memory provider**
backed by [HydraDB](https://hydradb.com): relevant context is recalled before
each turn and completed turns are persisted - the proper native way to make
HydraDB Hermes' long-term memory (like the supermemory / retaindb / ham providers).

## How it works

Implements Hermes' `MemoryProvider` contract:

| Method | Role → HydraDB |
|---|---|
| `prefetch(query, *, session_id)` | **auto-recall** - returns context injected before the API call → hybrid `POST /query` (prefers server `llm_prompt`) |
| `sync_turn(user, assistant, ...)` | persist the turn (**non-blocking**, off-thread) → `POST /context/ingest` |
| `is_available()` | `bool(HYDRADB_API_KEY)` |
| `get_config_schema()` / `save_config()` | setup wizard (api key, tenant) |
| `name` | `"hydradb"` |

Public API only (recall + knowledge ingest); no HydraDB SDK required.

## Prerequisites

- Python >= 3.10, a Hermes Agent install
- A HydraDB account: API key + tenant ID ([hydradb.com](https://hydradb.com))

## Install

**Directory (drop-in):** copy the `hydradb_hermes/` folder into
`~/.hermes/plugins/memory/hydradb/` (it contains `__init__.py` + `plugin.yaml`).

**pip (entry point):** `pip install hydradb-hermes` - registered via the
`hermes_agent.memory_providers` entry point.

Then enable it in `config.yaml` (memory providers are single-select):

```yaml
memory:
  provider: hydradb
```

Credentials come from the environment:

```bash
export HYDRADB_API_KEY="your-api-key"
export HYDRADB_DATABASE="your-tenant-id"
```

## Test

```bash
pip install -e ".[test]"
pytest
```

## MCP alternative

For pull-style recall (agent-invoked tools) instead of automatic memory, point
Hermes at the MCP server (see `hermes-mcp.snippet.json`). The provider gives
automatic recall/retain, so MCP is optional.

## License

Apache-2.0 - Copyright (c) 2026 HydraDB

### Structured query responses

The direct HTTP client keeps `query()` backward compatible: it returns a list
of chunks. Pass optional `include_table_sql` and `table_limit` to either
`query()` or `query_result()`. Omitted options are not sent.

```python
result = client.query_result("What is the total?", include_table_sql=True, table_limit=10)
chunks = result.chunks  # each chunk also exposes chunk_metadata
table = result.tabular_data
warnings = result.warnings  # includes any affected collections
temporal_facts = result.data.get("temporal_facts", [])  # observed_at is Unix seconds
graph = result.data.get("graph")  # graph data and provenance, when returned
```

`result.data` and `result.meta` preserve the response objects, including
temporal duration, graph context and future fields. Use them when the API
returns additional structured information; no extra request is made.
