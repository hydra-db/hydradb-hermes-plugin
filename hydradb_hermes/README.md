# HydraDB memory provider for Hermes

Long-term memory for [Hermes Agent](https://hermes-agent.nousresearch.com/docs/),
backed by [HydraDB](https://hydradb.com). Context is recalled before each turn and
completed turns are persisted, over the public HydraDB API.

## How it works

Registers a `MemoryProvider`:

| Method | Role |
|---|---|
| `prefetch(query)` | recall relevant context before a turn via `POST /query` (uses the server `llm_prompt` when present) |
| `sync_turn(user, assistant)` | persist the completed turn off-thread via `POST /context/ingest` |

No tools, hooks, or middleware. Public API only; no HydraDB SDK required.

## Setup

Run `hermes memory setup` and pick `hydradb`, or set the environment directly:

```bash
export HYDRADB_API_KEY="your-api-key"
export HYDRADB_TENANT_ID="your-tenant-id"
# optional: HYDRADB_SUB_TENANT_ID, HYDRADB_BASE_URL
```

Then enable it:

```yaml
memory:
  provider: hydradb
```

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `HYDRADB_API_KEY` | yes | HydraDB API key (secret) |
| `HYDRADB_TENANT_ID` | yes | database/tenant to read and write |
| `HYDRADB_SUB_TENANT_ID` | no | sub-tenant scope |
| `HYDRADB_BASE_URL` | no | defaults to `https://api.hydradb.com` |

The provider reports unavailable until `HYDRADB_API_KEY` and `HYDRADB_TENANT_ID`
are both set.

## License

Apache-2.0, Copyright (c) 2026 HydraDB
