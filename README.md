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

## Unified rollout preparation

Unified support is not enabled in production. Keep the default split layout
until your database's production rollout is confirmed. Then explicitly set
`HYDRADB_LAYOUT=unified` (or pass `layout="unified"` to `HydraDBClient`).
Only `split` and `unified` are accepted; failures never trigger a layout retry.

Unified writes send JSON `context[]`, with conversation turns as
`{role, content}`. Text-item IDs become `context_id`, `infer` becomes `enrich`,
`metadata` becomes `attributes`, and `additional_metadata` becomes
`custom_attributes`. The raw `add_knowledge` adapter supports text-only items
with those fields and `title`; it rejects unsupported fields instead of losing
them. Split ingestion remains unchanged.

`query()` still returns chunks; `query_result()` additionally retains the complete
response in `.data` and envelope metadata in `.meta`, including root `graph[]`,
`forceful_relations[]`, `llm_prompt`, and `tabular_data`. Unified chunks expose
their chunk ID as `.id` and their separate context ID as `.context_id`, alongside
`.enrichment` and `.received_at`; source titles/metadata are not fabricated.
The provider injects a supplied `llm_prompt` verbatim and persists turns off-thread.

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
