# Local execution

Run the Document Intelligence **AI service** on your machine, against the
Postgres container the backend API repo already runs. No Cloud Run, no Cloud SQL.

```bash
./setup_local.sh        # once — verify infra, reconcile schema, generate .env.local
./ai_server.sh start    # run it in the background
./ai_server.sh stop     # stop it
```

`./start_local.sh` runs the same server in the foreground instead, when you want
the logs in front of you and Ctrl+C to stop it.

---

## What this does and does not own

This service owns **no infrastructure**. `doc_intelligence_backend_api_v2.0`
runs `b2blocal-postgres`, which holds the shared `biz2bricks_core` schema, and
both services are meant to read the same database. These scripts join that
stack; they do not start a second Postgres.

That is why there is no `docker-compose.yml` here and no `stop_local.sh`. Stop
the containers where they are defined:

```bash
../../../doc_intelligence_backend_api_v2.0/scripts/local_exec/stop_infra.sh
```

If the backend repo lives somewhere else, set `B2B_BACKEND_REPO`.

| Script | Purpose |
|---|---|
| `setup_local.sh` | Verifies `b2blocal-postgres` is healthy (starts it via the backend's `setup_infra.sh` if not), asserts pgvector, applies any tables this repo's models declare that are missing, seeds tiers if empty, generates `.env.local`. Idempotent. |
| `start_local.sh` | Runs uvicorn on the host, in the foreground, wired to that database. Hot reload on by default. |
| `ai_server.sh` | `start` / `stop` / `restart` / `status` / `logs` for the same server, detached. A thin wrapper — all the environment and preflight logic stays in `start_local.sh`. |

### Running it in the background

```bash
./ai_server.sh start                      # detached; waits for /health before reporting success
./ai_server.sh start --port 8002 --no-reload
./ai_server.sh status                     # pid, port, component health, backend and Postgres too
./ai_server.sh logs -f                    # follow; -n N for a different tail length
./ai_server.sh restart
./ai_server.sh stop
```

State lives in `.run/` (pid, port, log), which is gitignored.

`stop` stops **only this service**. `b2blocal-postgres` is shared with the
backend API on :8000, so stopping it would break that session; the script says
so and points at the backend's `stop_infra.sh`.

Two details worth knowing:

- **`start` waits for readiness.** A server that dies on a missing env file or a
  rotated password reports the failure with the tail of its log, rather than
  claiming success on a process that is already gone.
- **`stop` sweeps by port, not by process name.** With `--reload`, uvicorn runs a
  supervisor plus a worker, and killing the supervisor hard can leave the worker
  holding the port. Sweeping the port catches that without any risk of killing
  an unrelated `uvicorn` — including the backend's. A port held by something
  this script did not start is reported, never killed.

`status` exits 0 when the service answers `/health` and 1 otherwise, so it works
in a conditional.

## Ports

```
AI service (this repo)   http://127.0.0.1:8001    /docs is live
Backend API              http://127.0.0.1:8000
Postgres                 127.0.0.1:15432          db doc_intelligence
```

8001 because the backend holds 8000. Override with `--port`.

## Credentials

`setup_local.sh` generates `.env.local` from `../../.env.local-gcp`, stripping
every Cloud SQL and database setting — those are exactly what local mode
replaces. Your API keys, model choices and GCS settings carry over unchanged.
It is gitignored and mode 600.

**It deliberately holds no database password.** `start_local.sh` reads the
Postgres credentials live from the backend repo's `.env.local` on every start.
Copying them would mean a password rotated over there leaves this service
failing to connect with a stale value and no obvious cause.

Point `SEED_ENV` at a different file if your keys live elsewhere.

## Where documents live

Object storage is **not** local. Parsed documents and generated content go to
the real bucket — `gs://biz2bricks-dev-v1-document-store` — the same one the
backend and the self-hosted stack use. Bucket storage is cheap and billed per
use; there is nothing to save by emulating it, and using the real bucket means
local runs see the same documents as deployed ones.

Raw uploads through `POST /api/v1/ingest/upload` are the exception: they land
in `./upload/{org_id}/` on disk, per this repo's design. Only `/parse` output
and generated content reach GCS. That matches the deployed behaviour, where
`/upload` is the container's ephemeral disk.

`start_local.sh` uses the service-account key at
`../biz2bricks_stack/secrets/gcp-sa-key.json`; override with `GCP_SA_KEY_FILE`
in `.env.local`. A key rather than your personal `gcloud` ADC on purpose: ADC
cannot sign URLs without an extra `iam.serviceAccounts.signBlob` grant, so the
signed upload URLs that bulk processing hands out would fail even while plain
reads and writes worked. Without a key the script falls back to ADC and says
so; without either it refuses to start.

## Calling the API

Both headers are required. `get_org_id` in `src/api/dependencies.py` rejects an
organization with no user attached, so org alone gives a 400, not a 403.

```bash
curl http://127.0.0.1:8001/api/v1/ingest/files \
  -H 'X-Organization-ID: <org-uuid>' \
  -H 'X-User-Email: <user-email>'
```

`./setup_local.sh` prints a working pair from your database when it finishes.

## What does not work locally

- **Nothing, as far as the service is concerned.** Every agent boots, and GCS,
  Gemini File Search, LlamaParse and OpenAI are reached over the network exactly
  as in production. They are billed per call, so local runs still cost what they
  consume.
- **Redis is not involved.** Nothing under `src/` imports it. The
  `b2blocal-redis` container belongs to the backend API.

## Useful commands

```bash
./setup_local.sh --status                  # containers, table/tier/org counts
./setup_local.sh --migrate-agent-builder   # opt-in: agent_definitions et al.
./ai_server.sh status                      # is the service up?
./start_local.sh --port 8002 --no-reload   # foreground
docker exec -it b2blocal-postgres psql -U postgres -d doc_intelligence
```

`--migrate-agent-builder` is opt-in because nothing under `src/` reads those
tables — only `scripts/apply_agent_builder_schema.py` writes them.

## Cost

These scripts remove the *need* for Cloud Run and Cloud SQL in day-to-day
development. They do not remove the *spend*: a Cloud SQL instance bills by the
hour whether or not anything connects to it. To actually save money, stop the
instance and drop Cloud Run to zero min-instances yourself — `scripts/gcp/teardown.sh`
covers that ground.
