# Document Intelligence AI v3.0

AI-powered document analysis system with Excel/CSV analysis and document content generation (summaries, FAQs, questions).

## Architecture

```
                        ┌─────────────────────────────────────────────────────┐
                        │              DOCUMENT INTELLIGENCE AI               │
                        └─────────────────────────────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                   API LAYER                     │
                        │  ┌───────────────────────────────────────────┐  │
                        │  │           FastAPI Application             │  │
                        │  │  ┌─────────┬────────┬───────┬──────────┐  │  │
                        │  │  │Documents│ Sheets │  RAG  │Ingest/   │  │  │
                        │  │  │ Router  │ Router │Router │Audit     │  │  │
                        │  │  └────┬────┴───┬────┴───┬───┴──────────┘  │  │
                        │  └───────┼────────┼────────┼─────────────────┘  │
                        │  ┌───────┴────────┴────────┴───────┐            │
                        │  │   Middleware (CORS, Auth, Log)  │            │
                        │  └─────────────────────────────────┘            │
                        └────────────────────────┬────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                  AGENT LAYER                    │
                        │                                                 │
                        │  ┌─────────────────┐    ┌─────────────────┐    │
                        │  │  DocumentAgent  │    │   SheetsAgent   │    │
                        │  │  (OpenAI GPT)   │    │  (OpenAI GPT)   │    │
                        │  │                 │    │                 │    │
                        │  │ Tools:          │    │ Tools:          │    │
                        │  │ -DocLoader      │    │ -SmartAnalysis  │    │
                        │  │ -Summary        │    │ -FilePreview    │    │
                        │  │ -FAQ            │    │ -SingleQuery    │    │
                        │  │ -Questions      │    │ -CrossQuery     │    │
                        │  │ -Persist        │    │ -DataAnalysis   │    │
                        │  │ -RAGSearch      │    │                 │    │
                        │  └────────┬────────┘    └────────┬────────┘    │
                        │           │                      │             │
                        │  ┌────────┴──────────────────────┴────────┐    │
                        │  │        Shared Infrastructure           │    │
                        │  │  ┌──────────┬────────────┬──────────┐  │    │
                        │  │  │RateLimiter│SessionMgr │ Memory   │  │    │
                        │  │  └──────────┴────────────┴──────────┘  │    │
                        │  │  ┌─────────────────────────────────┐   │    │
                        │  │  │ Middleware (Retry, PII, Tools)  │   │    │
                        │  │  └─────────────────────────────────┘   │    │
                        │  └────────────────────────────────────────┘    │
                        └────────────────────────┬────────────────────────┘
                                                 │
          ┌──────────────┬───────────────────────┼───────────────────────┬──────────────┐
          │              │                       │                       │              │
          ▼              ▼                       ▼                       ▼              ▼
   ┌────────────┐ ┌────────────┐ ┌────────────────────┐ ┌────────────┐ ┌────────────┐
   │ PostgreSQL │ │    GCS     │ │ Gemini File Search │ │   DuckDB   │ │ LlamaParse │
   │ (Cloud SQL)│ │ (Storage)  │ │ (Semantic Search)  │ │ (In-Memory)│ │ (Parsing)  │
   │            │ │            │ │                    │ │            │ │            │
   │-Orgs/Users │ │-Parsed docs│ │-Vector retrieval   │ │-SQL on     │ │-PDF/DOCX   │
   │-Documents  │ │-Generated  │ │-Hybrid search      │ │ DataFrames │ │-OCR        │
   │-Jobs/Audit │ │ content    │ │-Citations          │ │-Cross-file │ │-Handwriting│
   │-Memory     │ │-Uploads    │ │                    │ │ queries    │ │            │
   └────────────┘ └────────────┘ └────────────────────┘ └────────────┘ └────────────┘
```

**Agents:**

- **SheetsAgent** - Excel/CSV analysis with natural language queries (OpenAI gpt-5.1-codex-mini + DuckDB)
- **DocumentAgent** - Generate summaries, FAQs, and questions from documents (OpenAI gpt-4o-mini)
- **ExtractorAgent** - Structured data extraction with field analysis and schema-based extraction (OpenAI gpt-5-mini)

**Key Features:**

- Multi-tenancy with organization isolation
- **Bulk Processing** - Concurrent document processing with LangGraph state machine orchestration
- **Structured Extraction** - Field analysis and schema-based data extraction from documents
- **Gemini File Search** - Multi-tenant semantic search with hybrid modes (semantic/keyword/hybrid)
- GCS content caching with SHA-256 hash validation
- Session-based rate limiting and response caching
- Comprehensive middleware stack (retry, fallback, PII detection)

## Quick Start

### Prerequisites

- Python 3.12+
- Google Cloud SDK (for GCS authentication)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Authenticate with GCP (for GCS storage)
gcloud auth application-default login

# Create .env file with required keys
cat > .env << EOF
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
LLAMA_CLOUD_API_KEY=your-llama-key

# GCS Storage
GCS_BUCKET=biz2bricks-dev-v1-document-store
GCS_PREFIX=demo_docs
PARSED_DIRECTORY=parsed
GENERATED_DIRECTORY=generated
EOF
```

### Run

```bash
# Start the server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001

# Or run directly
python src/main.py

# Run test client (requires server running)
python src/app.py
```

API docs available at `/docs` (Swagger) and `/redoc` when server is running.

## Running Modes

Four supported setups:

| Mode | Database | Storage | Config file | How to run |
|------|----------|---------|-------------|------------|
| Local dev, no DB | disabled | GCS (or local) | `.env` | `uvicorn src.main:app --reload --port 8001` |
| **Local app → local Postgres** | `b2blocal-postgres` (Docker) | GCS (GCP) | `scripts/local_exec/.env.local` | `./scripts/local_exec/ai_server.sh start` |
| Local app → GCP | Cloud SQL (GCP) | GCS (GCP) | `.env.local-gcp` | `./scripts/start_local_gcp.sh` |
| Cloud Run (dev/prod) | Cloud SQL (GCP) | GCS (GCP) | Secret Manager + cloudbuild | `gcloud builds submit --config cloudbuild.yaml` |

The local-Postgres mode is the cheapest way to develop: it removes the need for
Cloud Run and Cloud SQL day to day, while documents stay in the real GCS bucket.

### Local Development (No Database)

```bash
# Add to .env
DATABASE_ENABLED=false
```

### Local App → GCP Resources (hybrid)

Runs the FastAPI server on your laptop but talks to the real Cloud SQL instance
and GCS buckets in `biz2bricks-dev-v1`. Useful for debugging prod data issues
without a full Cloud Run redeploy.

**One-time setup:**

```bash
# Install + log in to gcloud
gcloud auth login
gcloud auth application-default login
gcloud config set project biz2bricks-dev-v1
```

Application Default Credentials (ADC) are used by the Cloud SQL Python
Connector, the GCS client, and the Gemini / Vertex SDKs. The launcher bails
early if `~/.config/gcloud/application_default_credentials.json` is missing.

**Env file (`.env.local-gcp`):**

A pre-configured file already exists at the repo root. Key entries:

```dotenv
CLOUD_SQL_INSTANCE=biz2bricks-dev-v1:us-central1:doc-intelligence-db
DATABASE_NAME=doc_intelligence
DATABASE_USER=postgres
DATABASE_PASSWORD=<dev-password>
USE_CLOUD_SQL_CONNECTOR=true
CLOUD_SQL_IP_TYPE=PUBLIC           # laptop only; VPC deploys use PRIVATE
GCS_BUCKET=biz2bricks-dev-v1-document-store
```

Edit passwords / API keys in place. `.env*` is already in `.gitignore`.

**Start the server:**

```bash
# First run — also apply any pending schema migrations (e.g. agent_builder tables)
./scripts/start_local_gcp.sh --migrate

# Normal start
./scripts/start_local_gcp.sh

# Inspect the remote DB without starting the server
./scripts/start_local_gcp.sh --db-status

# Override port
PORT=8080 ./scripts/start_local_gcp.sh
```

The script:

1. Verifies gcloud ADC credentials exist
2. Sources `.env.local-gcp` and exports it
3. Activates `.venv`
4. Optionally runs `scripts/apply_agent_builder_schema.py` (`--migrate`)
5. Launches `uvicorn src.main:app --reload` on port 8001

API docs: http://localhost:8001/docs

### Local App → Local Postgres (no GCP spend)

Runs the FastAPI server and its database entirely on your laptop. Documents
still live in the real GCS bucket, exactly as the backend API does — object
storage is billed per use, not per hour, so there is nothing to save by
emulating it, and local runs then see the same documents as deployed ones.

**This service owns no infrastructure.** The backend API repo
(`doc_intelligence_backend_api_v2.0`) runs `b2blocal-postgres`, which holds the
shared `biz2bricks_core` schema. Both services read the same database, so these
scripts join that stack rather than starting a second Postgres. That is why
there is no compose file here.

**One-time setup:**

```bash
# Brings up b2blocal-postgres via the backend repo if it is not already running,
# asserts pgvector, reconciles the schema, seeds tiers, writes .env.local
./scripts/local_exec/setup_local.sh
```

It generates `scripts/local_exec/.env.local` from `.env.local-gcp`, stripping
every Cloud SQL and database setting — your API keys, model choices and GCS
settings carry over unchanged. The file deliberately holds **no database
password**: that is read live from the backend repo's `.env.local` on every
start, so a password rotated there cannot leave this service failing against a
stale copy.

Set `B2B_BACKEND_REPO` if the backend repo lives outside `../`.

**Run the server:**

```bash
# Background, with start/stop control
./scripts/local_exec/ai_server.sh start
./scripts/local_exec/ai_server.sh status     # pid, port, component health
./scripts/local_exec/ai_server.sh logs -f
./scripts/local_exec/ai_server.sh restart
./scripts/local_exec/ai_server.sh stop

# Foreground instead (Ctrl+C to stop)
./scripts/local_exec/start_local.sh
```

Port 8001, because the backend API holds 8000. Override with `--port`.

`ai_server.sh start` waits for `/health` before reporting success, so a server
that dies on a bad credential surfaces the error instead of a false green.
`stop` stops only this service — `b2blocal-postgres` is shared with the backend,
so it is left running; use the backend's `stop_infra.sh` to stop it too.

**Calling the API** — both headers are required, since `get_org_id` rejects an
organization with no user attached:

```bash
curl http://127.0.0.1:8001/api/v1/ingest/files \
  -H 'X-Organization-ID: <org-uuid>' \
  -H 'X-User-Email: <user-email>'
```

`setup_local.sh` prints a working pair from your database when it finishes.

Full detail: [`scripts/local_exec/README.md`](scripts/local_exec/README.md).

**On cost:** this removes the *need* for Cloud Run and Cloud SQL in development,
not the *spend*. A Cloud SQL instance bills by the hour whether or not anything
connects to it. To actually save money, stop the instance and drop Cloud Run to
zero min-instances — see `scripts/gcp/teardown.sh`.

### Production (Cloud SQL)

```bash
# Add to .env
DATABASE_ENABLED=true
CLOUD_SQL_INSTANCE=project:region:instance
DATABASE_NAME=doc_intelligence
DATABASE_USER=postgres
DATABASE_PASSWORD=your-password
USE_CLOUD_SQL_CONNECTOR=true
```

## Database Setup

```bash
# Create database and tables
python scripts/db_setup.py setup

# Show database status
python scripts/db_setup.py status

# Drop all tables
python scripts/db_setup.py teardown

# Reset database (teardown + setup)
python scripts/db_setup.py reset

# Print SQL schema without executing
python scripts/db_setup.py sql
```

### Schema migrations (agent_builder tables)

The Cloud SQL instance is shared with `agent_builder_v1`, which adds three
tables: `agent_definitions`, `agent_versions`, `agent_runs`. The migration is
tracked in `scripts/migrations/agent_builder_schema.sql` (idempotent `CREATE
TABLE IF NOT EXISTS`).

```bash
# Apply via the start script (preferred)
./scripts/start_local_gcp.sh --migrate

# Apply standalone (requires .env.local-gcp or .env_dev loaded)
python scripts/apply_agent_builder_schema.py

# Print the SQL without running it
python scripts/apply_agent_builder_schema.py --dry-run

# Verify the tables exist
python scripts/apply_agent_builder_schema.py --verify
```

## Testing

```bash
# Run all tests (unit + integration)
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test types
pytest tests/unit/                     # Unit tests only
pytest tests/integration/              # Integration tests only

# Integration tests require GCS credentials
RUN_INTEGRATION_TESTS=1 pytest tests/integration/
```

## API Endpoints

### Health

| Endpoint    | Method | Description  |
| ----------- | ------ | ------------ |
| `/health` | GET    | Health check |

### Documents (`/api/v1/documents/`)

| Endpoint          | Method | Description                         |
| ----------------- | ------ | ----------------------------------- |
| `/process`      | POST   | Process document with agent         |
| `/summarize`    | POST   | Generate summary (with GCS cache)   |
| `/faqs`         | POST   | Generate FAQs (with GCS cache)      |
| `/questions`    | POST   | Generate questions (with GCS cache) |
| `/generate-all` | POST   | Generate all content types          |
| `/chat`         | POST   | Conversational RAG with citations   |

### Sheets (`/api/v1/sheets/`)

| Endpoint     | Method | Description                |
| ------------ | ------ | -------------------------- |
| `/analyze` | POST   | Analyze Excel/CSV files    |
| `/preview` | POST   | Preview file contents      |
| `/health`  | GET    | Sheets agent health status |

### Ingestion (`/api/v1/ingest/`)

| Endpoint             | Method | Description                 |
| -------------------- | ------ | --------------------------- |
| `/upload`          | POST   | Upload files (org-scoped)   |
| `/parse`           | POST   | Parse documents (PDF, DOCX) |
| `/files`           | GET    | List uploaded files         |
| `/files/{file_id}` | GET    | Get file details            |
| `/files/{file_id}` | DELETE | Delete file                 |

### RAG (`/api/v1/rag/`) - Store Management

| Endpoint                      | Method | Description              |
| ----------------------------- | ------ | ------------------------ |
| `/stores`                   | POST   | Create file search store |
| `/stores`                   | GET    | List stores              |
| `/stores/{store_id}`        | GET    | Get store details        |
| `/stores/{store_id}/upload` | POST   | Upload document to store |
| `/stores/{store_id}/files`  | GET    | List files in store      |
| `/stores/{store_id}`        | DELETE | Delete store             |
| `/folders`                  | POST   | Create folder            |
| `/folders`                  | GET    | List folders             |
| `/folders/{folder_id}`      | GET    | Get folder details       |
| `/folders/{folder_id}`      | DELETE | Delete folder            |

#### Conversational RAG (via DocumentAgent)

Search is now consolidated into `/api/v1/documents/chat` endpoint, providing conversational RAG with session memory.

**Search Modes:**

- `semantic` - Vector similarity search (default)
- `keyword` - BM25 keyword matching
- `hybrid` - Combined semantic + keyword search

**Search Scopes:**

| Scope       | Parameters                         | Description                         |
| ----------- | ---------------------------------- | ----------------------------------- |
| Single File | `file_filter: "invoice.pdf"`     | Search within one specific file     |
| Folder      | `folder_filter: "Invoices 2024"` | Search all files in a folder        |
| Org-wide    | No filters                         | Search ALL indexed documents in org |

**Request Example** (`POST /api/v1/documents/chat`):

```json
{
  "query": "What are the payment terms?",
  "organization_name": "Acme Corp",
  "session_id": "sess_abc123",
  "folder_filter": "Legal",
  "file_filter": null,
  "search_mode": "hybrid",
  "max_sources": 5
}
```

**Response Example:**

```json
{
  "success": true,
  "answer": "Based on the documents, payment terms are Net 30...",
  "citations": [
    {
      "file": "contract.md",
      "text": "Payment shall be due within 30 days...",
      "relevance_score": 0.92,
      "folder_name": "Legal"
    }
  ],
  "session_id": "sess_abc123",
  "processing_time_ms": 1250,
  "search_mode": "hybrid"
}
```

### Audit (`/api/v1/audit/`)

| Endpoint                | Method | Description                                              |
| ----------------------- | ------ | -------------------------------------------------------- |
| `/dashboard`          | GET    | Dashboard statistics with period filter (7d/30d/90d/all) |
| `/activity`           | GET    | Activity timeline with recent events                     |
| `/jobs`               | GET    | List processing jobs (paginated)                         |
| `/jobs/{job_id}`      | GET    | Get job details                                          |
| `/documents`          | GET    | List processed documents                                 |
| `/documents/{doc_id}` | GET    | Get document record                                      |
| `/generations`        | GET    | List generated content                                   |
| `/trail`              | GET    | Audit trail with filtering                               |

### Bulk Processing (`/api/v1/bulk/`)

| Endpoint                      | Method | Description                                     |
| ----------------------------- | ------ | ----------------------------------------------- |
| `/upload`                   | POST   | Upload multiple files and start bulk processing |
| `/folders`                  | POST   | Create bulk folder                              |
| `/folders`                  | GET    | List org folders (paginated)                    |
| `/folders/{folder_id}`      | GET    | Get folder details                              |
| `/folders/{folder_id}`      | DELETE | Delete folder                                   |
| `/jobs`                     | GET    | List bulk jobs (paginated)                      |
| `/jobs/{job_id}`            | GET    | Get job details with progress                   |
| `/jobs/{job_id}/cancel`     | POST   | Cancel bulk job                                 |
| `/jobs/{document_id}/retry` | POST   | Retry failed document                           |

### Extraction (`/api/v1/extraction/`)

| Endpoint                       | Method | Description                             |
| ------------------------------ | ------ | --------------------------------------- |
| `/analyze-fields`            | POST   | Discover extractable fields in document |
| `/generate-schema`           | POST   | Generate extraction template            |
| `/extract`                   | POST   | Extract structured data using schema    |
| `/templates`                 | GET    | List extraction templates               |
| `/templates/{template_name}` | GET    | Get template details                    |
| `/records`                   | GET    | List extracted records                  |
| `/records/{record_id}`       | GET    | Get extracted record details            |
| `/export`                    | POST   | Export extracted data (CSV, JSON)       |
| `/health`                    | GET    | Extraction service health status        |

### Usage (`/api/v1/usage/`)

| Endpoint          | Method | Description                     |
| ----------------- | ------ | ------------------------------- |
| `/summary`      | GET    | Current period usage summary    |
| `/subscription` | GET    | Subscription details and limits |
| `/limits`       | GET    | Quota status with warnings      |
| `/history`      | GET    | Historical usage data           |
| `/breakdown`    | GET    | Usage breakdown by feature      |

### Sessions (`/api/v1/sessions/`)

| Endpoint          | Method | Description             |
| ----------------- | ------ | ----------------------- |
| `/{session_id}` | GET    | Get session info        |
| `/{session_id}` | DELETE | End session and cleanup |

### Tiers (`/api/v1/tiers/`)

| Endpoint | Method | Description                               |
| -------- | ------ | ----------------------------------------- |
| `/`    | GET    | List subscription tiers (public, no auth) |

### Content (`/api/v1/content/`)

| Endpoint          | Method | Description                               |
| ----------------- | ------ | ----------------------------------------- |
| `/load-parsed`  | POST   | Load pre-parsed document content from GCS |
| `/check-exists` | GET    | Check if parsed document exists in GCS    |

## Core Features

### Multi-Tenancy

- Organization context extracted from `X-Organization-ID` header
- All data operations scoped by org_id
- Per-org file storage and store isolation

### Memory System

- **Short-term Memory**: Per-session conversation history (max 20 messages)
- **Long-term Memory**: PostgreSQL-backed persistent summaries and context

### Middleware Stack

Both agents use a composable middleware stack for resilience and safety:

- **Retry Logic**: Exponential backoff for model and tool calls (3 attempts)
- **Model Fallback**: Primary/fallback model switching
- **PII Detection**: Input/output redaction (redact/mask/hash/block strategies)
- **Call Limits**: Prevents runaway tool/model calls
- **Hybrid Tool Selection**: Two-stage intent-based routing
  - `QueryClassifier`: Rule-based + LLM intent detection (RAG vs Generation)
  - `LLMToolSelector`: Pre-filters tools based on query relevance

### Caching

- **File Cache**: LRU cache (50 files) for SheetsAgent
- **Response Cache**: Per-session query hash caching (10 responses)
- **GCS Cache**: SHA-256 content hash validation for document generation
- **Generation Cache**: PostgreSQL with content hash validation

### Rate Limiting

10 requests per 60-second window per session.

### Background Tasks

- Periodic cleanup task (expired sessions, rate limiter entries)
- Async audit logging via ThreadPoolExecutor (non-blocking)
- Token usage tracking and cost estimation

## Shared Infrastructure

### Utility Modules (`src/utils/`)

- **gcs_utils.py**: GCS path parsing (`is_gcs_path()`, `parse_gcs_uri()`, `build_gcs_uri()`, `extract_gcs_path_parts()`)
- **timer_utils.py**: Performance timing (`elapsed_ms()`, `Timer` context manager)
- **async_utils.py**: Async/sync interop (`run_async()`, `run_sync_in_executor()`)
- **env_utils.py**: Environment variable parsing (`parse_bool_env()`, `parse_int_env()`, `parse_float_env()`)

### Constants (`src/constants.py`)

Centralized configuration values replacing magic numbers:

- GCS URI prefix constants
- Cache sizes (file: 50, response: 10 per session)
- Session timeout (30 min), rate limits (10 req/60s)
- Agent timeout (300s), cleanup interval (300s)
- Database pool configuration (size: 5, max overflow: 10)
- Content generation defaults (10 FAQs, 10 questions, 500 word summary)

### Agent Infrastructure (`src/agents/core/`)

- **RateLimiter** (`rate_limiter.py`): Thread-safe sliding window rate limiting per session
- **SessionManager** (`session_manager.py`): Session lifecycle with response caching
- **Memory**: Short-term (in-memory, 20 messages) + long-term (PostgreSQL)
- **Middleware**: Retry logic, model fallback, PII detection, tool selection

## Environment Variables

### Required

```
OPENAI_API_KEY=<key>
GOOGLE_API_KEY=<key>
LLAMA_CLOUD_API_KEY=<key>
GCS_BUCKET=<bucket-name>
GCS_PREFIX=<prefix>
```

### Database (optional)

```
DATABASE_ENABLED=true
CLOUD_SQL_INSTANCE=<project>:<region>:<instance>
DATABASE_NAME=doc_intelligence
DATABASE_USER=postgres
DATABASE_PASSWORD=<password>
USE_CLOUD_SQL_CONNECTOR=true
```

### Agent Configuration

```
OPENAI_SHEET_MODEL=gpt-5.1-codex-mini
DOCUMENT_AGENT_MODEL=gpt-4o-mini
EXTRACTOR_AGENT_MODEL=gpt-5-mini
EXTRACTOR_FALLBACK_MODEL=gpt-5.2-2025-12-11
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60
SESSION_TIMEOUT_MINUTES=30
```

### Middleware & Tool Selection

```
ENABLE_MIDDLEWARE=true
MODEL_RETRY_MAX_ATTEMPTS=3
TOOL_RETRY_MAX_ATTEMPTS=2
ENABLE_PII_DETECTION=true
PII_STRATEGY=redact
ENABLE_TOOL_SELECTION=true
TOOL_SELECTOR_MODEL=gpt-5-mini
TOOL_SELECTOR_MAX_TOOLS=3
USE_DIRECT_INVOCATION=true  # Bypass ReAct agent for deterministic workflows
```

### Bulk Processing

```
BULK_MAX_DOCUMENTS_PER_FOLDER=10
BULK_CONCURRENT_DOCUMENTS=3
BULK_JOB_TIMEOUT_SECONDS=3600
BULK_WEBHOOK_SECRET=<secret>
```

### Extractor Agent

```
EXTRACTOR_AGENT_MODEL=gpt-5-mini
EXTRACTOR_FALLBACK_MODEL=gpt-5.2-2025-12-11
EXTRACTOR_MAX_FIELDS=50
EXTRACTOR_MAX_SCHEMA_FIELDS=100
EXTRACTOR_TIMEOUT_SECONDS=120
```

### Executor Pools

```
AGENT_EXECUTOR_POOL_SIZE=10
IO_EXECUTOR_POOL_SIZE=20
QUERY_EXECUTOR_POOL_SIZE=10
```

## Project Structure

```
src/
├── main.py               # Entry point
├── app.py                # Test client
├── constants.py          # Centralized application constants
├── utils/                # Utility modules
│   ├── gcs_utils.py      # GCS path parsing utilities
│   ├── timer_utils.py    # Performance timing utilities
│   ├── async_utils.py    # Async/sync helpers
│   └── env_utils.py      # Environment variable parsing
├── api/                  # FastAPI routers & schemas
│   ├── app.py            # App factory with lifespan management
│   ├── middleware.py     # CORS, logging, exception handlers
│   ├── dependencies.py   # Dependency injection
│   ├── usage.py          # Token usage tracking
│   ├── routers/          # Route handlers
│   │   ├── documents.py  # DocumentAgent endpoints
│   │   ├── sheets.py     # SheetsAgent endpoints
│   │   ├── extraction/   # ExtractorAgent endpoints (package)
│   │   │   ├── analyze.py    # Field analysis
│   │   │   ├── schema.py     # Schema generation
│   │   │   ├── extract.py    # Data extraction
│   │   │   ├── records.py    # Record management
│   │   │   ├── export.py     # Export functionality
│   │   │   ├── cache.py      # Extraction caching
│   │   │   ├── helpers.py    # Shared helpers
│   │   │   └── health.py     # Health endpoint
│   │   ├── bulk.py       # Bulk processing endpoints
│   │   ├── rag.py        # Semantic search endpoints
│   │   ├── ingest.py     # Upload/parse endpoints
│   │   ├── audit.py      # Audit trail + dashboard endpoints
│   │   ├── usage.py      # Usage/subscription endpoints
│   │   ├── sessions.py   # Session management
│   │   ├── tiers.py      # Subscription tiers (public)
│   │   ├── content.py    # Content loading from GCS
│   │   ├── health.py     # Health check endpoints
│   │   └── rag_helpers.py  # RAG helper utilities
│   ├── utils/            # API utilities
│   │   ├── formatting.py # Display formatting helpers
│   │   ├── decorators.py # Endpoint decorators
│   │   └── responses.py  # Response builders
│   └── schemas/          # Pydantic models
│       ├── errors.py     # Shared error response definitions
│       ├── common.py     # Shared schemas (TokenUsage, ToolUsage)
│       ├── validators.py # Shared validators (path, query, options)
│       ├── documents.py  # Document request/response
│       ├── sheets.py     # Sheets request/response
│       ├── extraction.py # Extraction request/response
│       ├── bulk.py       # Bulk processing request/response
│       └── rag.py        # RAG request/response
├── agents/
│   ├── core/             # Shared agent infrastructure
│   │   ├── base_agent.py      # Abstract base for all agents
│   │   ├── base_config.py     # Shared configuration
│   │   ├── rate_limiter.py    # Thread-safe rate limiting
│   │   ├── session_manager.py # Session lifecycle management
│   │   ├── memory/       # Conversation memory
│   │   │   ├── short_term.py  # In-memory (20 messages)
│   │   │   └── long_term.py   # PostgreSQL-backed
│   │   └── middleware/   # Agent middleware stack
│   │       ├── query_classifier.py  # Intent detection
│   │       ├── tool_selector.py     # Tool pre-filtering
│   │       ├── resilience.py        # Retry logic
│   │       └── safety.py            # PII detection
│   ├── document/         # DocumentAgent (OpenAI gpt-4o-mini)
│   │   ├── core.py       # Agent implementation (1786 lines)
│   │   ├── config.py     # Agent configuration
│   │   ├── schemas.py    # Request/response schemas
│   │   ├── gcs_cache.py  # GCS content caching
│   │   ├── tool_selection.py  # ToolSelectionManager + bind_rag_filters
│   │   ├── result_parser.py   # AgentResultParser for tool outputs
│   │   └── tools/        # Tool package
│   │       ├── __init__.py         # Tool factory
│   │       ├── base.py             # Shared utilities
│   │       ├── document_loader.py  # Load from GCS/local
│   │       ├── summary_generator.py
│   │       ├── faq_generator.py
│   │       ├── question_generator.py
│   │       ├── persist.py          # Save to GCS/DB
│   │       └── rag_search.py       # Semantic search
│   ├── extractor/        # ExtractorAgent (OpenAI gpt-5-mini)
│   │   ├── core.py       # Agent implementation (777 lines)
│   │   ├── config.py     # Agent configuration
│   │   ├── schemas.py    # Request/response schemas
│   │   └── tools/        # Extraction tools
│   │       ├── base.py             # Shared utilities
│   │       ├── field_analyzer.py   # Field discovery
│   │       ├── schema_generator.py # Template generation
│   │       └── data_extractor.py   # Data extraction
│   └── sheets/           # SheetsAgent (OpenAI + DuckDB)
│       ├── core.py       # Agent implementation (682 lines)
│       ├── config.py     # Agent configuration
│       ├── cache.py      # FileCache (LRU DataFrame caching)
│       └── tools.py      # Analysis tools
├── bulk/                 # Bulk document processing
│   ├── config.py         # BulkProcessingConfig
│   ├── schemas.py        # Job/document/event models
│   ├── service.py        # Main orchestration service
│   ├── state_graph.py    # LangGraph workflow
│   ├── generation.py     # Content generation helpers
│   ├── folder_manager.py # Folder operations
│   ├── queue.py          # Event queue
│   └── webhook_handler.py # Cloud Function webhooks
├── core/                 # Core infrastructure
│   ├── executors.py      # Thread pool registry
│   └── usage/            # Usage tracking module
│       ├── service.py    # UsageTrackingService
│       ├── quota_checker.py  # Quota enforcement
│       ├── usage_queue.py    # Background usage tracking
│       ├── context.py        # Usage context management
│       └── ...
├── services/             # Shared services
│   └── parse_service.py  # Document parsing abstraction
├── db/                   # PostgreSQL models & connection
│   ├── connection.py     # DatabaseManager singleton
│   ├── models.py         # SQLAlchemy ORM models
│   └── repositories/     # Data access layer
│       ├── audit/               # Split audit repositories
│       │   ├── document_repository.py
│       │   ├── job_repository.py
│       │   ├── audit_log_repository.py
│       │   ├── generation_repository.py
│       │   └── stats_repository.py  # Dashboard statistics
│       ├── bulk_repository.py   # Bulk job persistence
│       ├── extraction_repository.py # Extraction persistence
│       ├── memory_repository.py
│       ├── rag_repository.py
│       └── semantic_cache_repository.py  # Semantic cache persistence
├── rag/                  # Document parsing & search
│   ├── llama_parse_util.py    # LlamaParse integration
│   ├── gemini_file_store.py   # Gemini File Search API
│   └── file_search_service.py # High-level orchestration
└── storage/              # GCS storage abstraction
    ├── base.py           # StorageBackend interface
    ├── gcs.py            # GCSStorage implementation
    └── config.py         # Storage configuration

cloud_functions/
└── bulk_trigger/         # GCS-triggered bulk processing
    └── main.py           # Cloud Function entry point

scripts/
├── db_setup.py           # Database setup/teardown
├── seed_tiers.py         # Seed subscription tiers
├── migrate_orgs_free.py  # Migrate organizations to free tier
├── migrate_documents_status.py  # Data migration
├── clean_all_data.py     # Clean all data utility
└── fix_cache_relevance.py  # Cache relevance fixes

tests/
├── conftest.py           # Pytest fixtures
├── unit/                 # Unit tests
└── integration/          # Integration tests

# GCS Storage (gs://bucket/prefix/)
parsed/               # Pre-parsed documents (.md)
generated/            # Generated content output (.json)

# Local Storage
upload/               # User uploads (text files)
docs/                 # Sample documents
```

## Storage

Parsed documents and generated content are stored in Google Cloud Storage:

| Content     | GCS Path                                                      |
| ----------- | ------------------------------------------------------------- |
| Parsed docs | `gs://{GCS_BUCKET}/{GCS_PREFIX}/parsed/*.md`                |
| Generated   | `gs://{GCS_BUCKET}/{GCS_PREFIX}/generated/*_generated.json` |

Authentication uses Application Default Credentials (ADC).

## Dependencies

Key dependencies from `requirements.txt`:

- `biz2bricks-core` - Shared core models and utilities
- `langgraph>=1.0.4` - LangGraph for ReAct agents
- `langchain>=1.2.0` - LangChain framework
- `langchain-openai>=1.1.0` - OpenAI integration
- `langchain-google-genai>=3.2.0` - Gemini integration
- `llama-cloud-services>=0.6.88` - LlamaParse document parsing
- `google-genai>=1.53.0` - Gemini File Store API
- `fastapi>=0.115.0` - Web framework
- `sqlalchemy[asyncio]>=2.0.0` - Async ORM
- `duckdb>=0.9.0` - SQL on DataFrames
- `google-cloud-storage>=2.14.0` - GCS client

## GCP Deployment

This section covers deploying the AI API to Google Cloud Run.

### TL;DR — deploy dev

Assumes prerequisites (below) are done and secrets are already in Secret Manager.

```bash
# 1. Smoke-test locally against the same Cloud SQL + GCS the deploy will use
./scripts/start_local_gcp.sh
# hit http://localhost:8001/health, run through the endpoints you care about

# 2. Apply any pending schema migrations to Cloud SQL
./scripts/start_local_gcp.sh --migrate   # or: python scripts/apply_agent_builder_schema.py

# 3. Deploy to the dev Cloud Run service
gcloud builds submit --config cloudbuild.yaml

# 4. Verify
SERVICE_URL=$(gcloud run services describe document-intelligence-ai-api-dev \
  --region=us-central1 --format="value(status.url)")
curl "$SERVICE_URL/health"
```

For prod, swap the last step:

```bash
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_ENV=prod,_SERVICE_NAME=document-intelligence-ai-api-prod,_SECRET_SUFFIX=-prod
```

Full details below.

### Prerequisites

- **Google Cloud SDK** installed and configured
- **Docker** installed locally
- **GCP Project** with billing enabled
- Cloud SQL instance provisioned
- GCS bucket created

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Set up Application Default Credentials
gcloud auth application-default login

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Required GCP APIs

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  sqladmin.googleapis.com \
  secretmanager.googleapis.com \
  storage.googleapis.com
```

### Secrets Configuration

Create secrets in Secret Manager using the provided script:

```bash
# Create all required secrets
./scripts/gcp/create_secrets.sh
```

Or create them manually:

```bash
# OpenAI API Key
echo -n "sk-your-openai-key" | gcloud secrets create openai-api-key --data-file=-

# Google API Key (for Gemini)
echo -n "your-google-api-key" | gcloud secrets create google-api-key --data-file=-

# LlamaParse API Key
echo -n "your-llamaparse-key" | gcloud secrets create llamaparse-api-key --data-file=-

# Database Password
echo -n "your-db-password" | gcloud secrets create DATABASE_PASSWORD --data-file=-

# For production (with -prod suffix)
echo -n "sk-prod-key" | gcloud secrets create openai-api-key-prod --data-file=-
echo -n "prod-google-key" | gcloud secrets create google-api-key-prod --data-file=-
echo -n "prod-llama-key" | gcloud secrets create llamaparse-api-key-prod --data-file=-
echo -n "prod-db-pass" | gcloud secrets create DATABASE_PASSWORD-prod --data-file=-
```

Grant Cloud Run access to secrets:

```bash
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
for SECRET in openai-api-key google-api-key llamaparse-api-key DATABASE_PASSWORD; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
done
```

### Manual Deployment

```bash
# Build and deploy to development
gcloud builds submit --config cloudbuild.yaml

# Deploy to production
gcloud builds submit --config cloudbuild.yaml \
  --substitutions=_ENV=prod,_SERVICE_NAME=document-intelligence-ai-api-prod,_SECRET_SUFFIX=-prod

# Or use the deployment script
./scripts/gcp/deploy_to_cloud_run.sh
```

### CI/CD Triggers

Set up automatic deployments from GitHub:

```bash
# Run the setup script
./scripts/gcp/setup_triggers.sh
```

**Trigger Configuration:**

| Trigger | Branch | Service Name | Secret Suffix |
|---------|--------|--------------|---------------|
| `ai-api-dev-deploy` | `develop` | `document-intelligence-ai-api-dev` | (none) |
| `ai-api-prod-deploy` | `master` | `document-intelligence-ai-api-prod` | `-prod` |

### Cloud Run Service Details

| Setting | Development | Production |
|---------|-------------|------------|
| Service Name | `document-intelligence-ai-api-dev` | `document-intelligence-ai-api-prod` |
| Port | 8080 | 8080 |
| Memory | 2Gi | 2Gi |
| CPU | 2 | 2 |
| Min Instances | 0 | 0 |
| Max Instances | 5 | 10 |
| Timeout | 300s | 300s |
| Cloud SQL | Connected via Auth Proxy | Connected via Auth Proxy |

### Environment Variables

Automatically configured via `cloudbuild.yaml`:

| Variable | Description |
|----------|-------------|
| `CLOUD_SQL_INSTANCE` | Cloud SQL connection name |
| `DATABASE_NAME` | Database name |
| `DATABASE_USER` | Database user |
| `DATABASE_ENABLED` | Enable database operations |
| `USE_CLOUD_SQL_CONNECTOR` | Use Cloud SQL Auth Proxy |
| `GCS_BUCKET` | GCS bucket for storage |
| `GCS_PREFIX` | GCS path prefix |
| `PARSED_DIRECTORY` | Directory for parsed documents |
| `GENERATED_DIRECTORY` | Directory for generated content |
| `LOG_LEVEL` | Logging level (DEBUG/INFO) |

### Required IAM Roles

The Cloud Run service account needs:

| Role | Purpose |
|------|---------|
| `roles/cloudsql.client` | Connect to Cloud SQL |
| `roles/storage.objectAdmin` | Access GCS bucket |
| `roles/logging.logWriter` | Write logs |
| `roles/secretmanager.secretAccessor` | Access API keys and secrets |

### Verification

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe document-intelligence-ai-api-dev \
  --region=us-central1 --format="value(status.url)")

# Test health endpoint
curl "$SERVICE_URL/health"
# Expected: {"status": "healthy", ...}

# Test a simple API call (requires X-Organization-ID header)
curl -H "X-Organization-ID: test-org" "$SERVICE_URL/api/v1/tiers"
```

### Rollback

```bash
# List recent revisions
gcloud run revisions list \
  --service=document-intelligence-ai-api-dev \
  --region=us-central1

# Rollback using the script
./scripts/gcp/rollback.sh document-intelligence-ai-api-dev REVISION_NAME

# Or manually route traffic
gcloud run services update-traffic document-intelligence-ai-api-dev \
  --region=us-central1 \
  --to-revisions=REVISION_NAME=100
```

### Troubleshooting

**Health check failing (503)?**
```bash
# Check container logs
gcloud run services logs read document-intelligence-ai-api-dev \
  --region=us-central1 --limit=50

# Verify Cloud SQL connectivity
gcloud sql instances describe doc-intelligence-db --format="value(connectionName)"
```

**Secret access denied?**
```bash
# Verify secrets exist
gcloud secrets list --filter="name~api-key"

# Check IAM binding
gcloud secrets get-iam-policy openai-api-key

# Verify secret value (careful - shows actual value)
gcloud secrets versions access latest --secret=openai-api-key
```

**OpenAI/Gemini API errors?**
```bash
# Check recent logs for API errors
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=document-intelligence-ai-api-dev AND \
  textPayload=~'API'" \
  --limit=20

# Verify environment variables
gcloud run services describe document-intelligence-ai-api-dev \
  --region=us-central1 --format="yaml(spec.template.spec.containers[0].env)"
```

**GCS access errors?**
```bash
# Verify bucket exists
gsutil ls gs://YOUR_BUCKET_NAME

# Check service account has storage permissions
gcloud projects get-iam-policy $PROJECT_ID \
  --filter="bindings.members:compute@developer" \
  --format="table(bindings.role)" | grep storage
```

### GCP Scripts Reference

| Script | Purpose |
|--------|---------|
| `scripts/gcp/config.sh` | Shared configuration variables |
| `scripts/gcp/create_secrets.sh` | Create Secret Manager secrets |
| `scripts/gcp/deploy_to_cloud_run.sh` | Manual deployment script |
| `scripts/gcp/setup_triggers.sh` | Set up Cloud Build triggers |
| `scripts/gcp/rollback.sh` | Rollback to previous revision |
