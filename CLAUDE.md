# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Document Intelligence AI v3.0 is an AI-powered document analysis system with three main agents:
- **SheetsAgent**: Excel/CSV analysis using OpenAI GPT and LangGraph's ReAct agent
- **DocumentAgent**: Document processing with content generation (summaries, FAQs, questions) using OpenAI GPT and LangChain
- **ExtractorAgent**: Structured data extraction from documents using OpenAI GPT with field analysis and schema-based extraction

## Technology Stack

- Python 3.12, LangGraph 1.0.4+, LangChain 1.2.0+
- LLMs: OpenAI (gpt-5.1-codex-mini for SheetsAgent, gpt-5-mini for DocumentAgent, gpt-5-mini for ExtractorAgent)
- Data: DuckDB (SQL on DataFrames with connection pooling), Pandas, LlamaParse (document parsing with OCR)
- Storage: Google Cloud Storage (parsed docs, generated content), PostgreSQL (Cloud SQL), SQLAlchemy 2.0 async
- Web Framework: FastAPI with uvicorn
- Testing: pytest, pytest-asyncio, pytest-cov

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Start FastAPI server (main entry point)
uvicorn src.main:app --reload --host 0.0.0.0 --port 8001

# Or run directly
python src/main.py

# Run Gemini File Store test client (parse PDF -> upload -> search)
python src/app.py

# Install dependencies
pip install -r requirements.txt

# Database setup
python scripts/db_setup.py setup      # Create database and tables
python scripts/db_setup.py status     # Show current database state
python scripts/db_setup.py teardown   # Drop all tables
python scripts/db_setup.py reset      # Teardown + setup
python scripts/db_setup.py sql        # Print all SQL without executing

# Seed subscription tiers
python scripts/seed_tiers.py          # Seed Free/Pro/Enterprise tiers
python scripts/seed_tiers.py --list   # List current tiers
python scripts/seed_tiers.py --reset  # Delete and re-seed tiers

# Run tests
pytest tests/                          # Run all tests (unit + integration)
pytest tests/ --cov=src                # Run with coverage
pytest tests/unit/                     # Run unit tests only
pytest tests/integration/              # Run integration tests only
pytest tests/unit/storage/test_gcs_storage.py  # Run single test file
pytest tests/unit/storage/test_gcs_storage.py::test_save_file -v  # Run single test function

# Integration tests require GCS credentials
RUN_INTEGRATION_TESTS=1 pytest tests/integration/
```

API docs available at `/docs` (Swagger) and `/redoc` when server is running.

## Required Environment Variables

```
# API Keys
OPENAI_API_KEY=<key>
GOOGLE_API_KEY=<key>  # Required for Gemini File Search (RAG module)
LLAMA_CLOUD_API_KEY=<key>

# Database (Cloud SQL)
CLOUD_SQL_INSTANCE=<project>:<region>:<instance>
DATABASE_NAME=doc_intelligence
DATABASE_USER=postgres
DATABASE_PASSWORD=<password>

# GCS Storage (required)
GCS_BUCKET=biz2bricks-dev-v1-document-store
GCS_PREFIX=demo_docs
PARSED_DIRECTORY=parsed
GENERATED_DIRECTORY=generated

# Optional - Database
DATABASE_ENABLED=true          # Set false to skip DB operations
USE_CLOUD_SQL_CONNECTOR=true   # Set false for direct connection
DATABASE_URL=postgresql+asyncpg://...  # For local development

# Optional - Agents
OPENAI_SHEET_MODEL=gpt-5.1-codex-mini
DOCUMENT_AGENT_MODEL=gpt-5-mini
OPENAI_TEMPERATURE=0.1
DOCUMENT_AGENT_TEMPERATURE=0.3

# Optional - Extractor Agent
EXTRACTOR_AGENT_MODEL=gpt-5-mini
EXTRACTOR_FALLBACK_MODEL=gpt-5.2-2025-12-11
EXTRACTOR_AGENT_TEMPERATURE=0.2
EXTRACTOR_MAX_FIELDS=50
EXTRACTOR_MAX_SCHEMA_FIELDS=100
EXTRACTOR_TIMEOUT_SECONDS=120

# Optional - Rate Limiting & Sessions
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60
SESSION_TIMEOUT_MINUTES=30

# Optional - Middleware
ENABLE_MIDDLEWARE=true
MODEL_RETRY_MAX_ATTEMPTS=3
TOOL_RETRY_MAX_ATTEMPTS=2
ENABLE_PII_DETECTION=true
PII_STRATEGY=redact  # redact|mask|hash|block

# Optional - Memory
ENABLE_SHORT_TERM_MEMORY=true
ENABLE_LONG_TERM_MEMORY=true
SHORT_TERM_MEMORY_MAX_MESSAGES=20

# Optional - General
LOG_LEVEL=INFO
DEBUG=false
CLEANUP_INTERVAL_SECONDS=300

# Optional - Bulk Processing
BULK_MAX_DOCUMENTS_PER_FOLDER=10
BULK_MAX_FILE_SIZE_MB=50
BULK_CONCURRENT_DOCUMENTS=3
BULK_PARSE_TIMEOUT_SECONDS=300
BULK_GENERATION_TIMEOUT_SECONDS=300
BULK_JOB_TIMEOUT_SECONDS=3600
BULK_MAX_RETRIES_PER_DOCUMENT=3
BULK_RETRY_DELAY_SECONDS=30
BULK_WEBHOOK_SECRET=<secret>
BULK_USE_POSTGRES_CHECKPOINTER=true

# Optional - Executor Pool Sizes
AGENT_EXECUTOR_POOL_SIZE=10   # Thread pool for LLM agent invocations
IO_EXECUTOR_POOL_SIZE=20      # Thread pool for GCS/file I/O operations
QUERY_EXECUTOR_POOL_SIZE=10   # Thread pool for DuckDB/SQL queries
```

**GCS Authentication**: Uses Application Default Credentials (ADC). Run `gcloud auth application-default login` for local development.

## Architecture

### API Layer (`src/api/`)

FastAPI application with routers:
- `/api/v1/documents` - DocumentAgent endpoints (process, summarize, faqs, questions, generate-all)
- `/api/v1/sheets` - SheetsAgent endpoints (analyze, preview, health)
- `/api/v1/extraction` - ExtractorAgent endpoints (analyze-fields, generate-schema, extract, records, export)
- `/api/v1/bulk` - Bulk processing endpoints (upload, folders, jobs, cancel, retry)
- `/api/v1/ingest` - Document ingestion (upload, parse, list, delete)
- `/api/v1/rag` - Semantic search (stores, folders, search, upload)
- `/api/v1/audit` - Audit logs (dashboard, activity, jobs, documents, generations, trail)
- `/api/v1/usage` - Usage tracking (summary, subscription, limits, history, breakdown)
- `/api/v1/sessions` - Session management (get, delete)
- `/api/v1/tiers` - Subscription tiers (public, no auth required)
- `/api/v1/content` - Content loading (load-parsed, check-exists)
- `/health` - Health checks

Entry point: `src/main.py` creates app via `src/api/app.py:create_app()`

Key files:
- `app.py`: FastAPI app factory with lifespan management (startup/shutdown, periodic cleanup)
- `middleware.py`: CORS, logging, exception handlers
- `dependencies.py`: Dependency injection for agents, org context, file directories
- `usage.py`: Token usage tracking and cost estimation
- `utils/`: API utility package
  - `formatting.py`: Display formatting (`format_duration_ms`, `format_time_ago`, `get_status_color`)
  - `decorators.py`: Endpoint decorators (`with_timing`, `with_error_response`, `log_errors`)
  - `responses.py`: Response builders (`build_success_response`, `ResponseBuilder`)

### Agents (`src/agents/`)

**SheetsAgent** (`src/agents/sheets/core.py`, 686 lines):
- ReAct pattern with `langgraph.prebuilt.create_react_agent`
- Uses `ChatOpenAI` with `use_responses_api=True` for gpt-5.1-codex-mini compatibility
- Tools in `tools.py`:
  - `SmartAnalysisTool`: Combined preview + query analysis (primary tool for single files)
  - `FilePreviewTool`: File structure and metadata preview
  - `SingleFileQueryTool`: Natural language queries on single file
  - `CrossFileQueryTool`: SQL queries across multiple files via DuckDB
  - `DataAnalysisTool`: Statistical analysis (summary, correlation, trends, outliers, quality)
- DuckDB connection pool (`DuckDBPool`) for SQL queries on DataFrames
- SQL injection protection via `validate_sql_query()` and `DANGEROUS_SQL_KEYWORDS`
- LRU file cache via `FileCache` class (`cache.py`) with 50-file capacity

**DocumentAgent** (`src/agents/document/core.py`, 875 lines):
- Uses `init_chat_model` from `langchain.chat_models` for OpenAI LLM initialization
- Tools organized as package (`tools/`):
  - `base.py`: Shared utilities (path derivation, content formatting, input schemas)
  - `document_loader.py`: Load documents from GCS parsed directory or local upload
  - `summary_generator.py`: Generate summaries with configurable word limit
  - `faq_generator.py`: Generate FAQs with JSON output format
  - `question_generator.py`: Generate questions (easy/medium/hard difficulty)
  - `persist.py`: Save generated content to GCS and PostgreSQL
  - `rag_search.py`: Conversational RAG with Gemini File Search and citations
- Factory: `create_document_tools(config)` in `tools/__init__.py` creates all tools
- GCS caching (`gcs_cache.py`) with SHA-256 hash validation via `check_and_read_cached_*()` functions
- Convenience methods: `generate_summary()`, `generate_faqs()`, `generate_questions()`, `generate_all()`, `chat()`
- Hybrid tool selection via `ToolSelectionManager` (`tool_selection.py`) with `QueryClassifier` + `LLMToolSelector`
- Result parsing via `AgentResultParser` (`result_parser.py`) for extracting structured content from tool outputs

**ExtractorAgent** (`src/agents/extractor/core.py`, 734 lines):
- Structured data extraction from documents using OpenAI gpt-5-nano
- Three main capabilities:
  - **Field Analysis**: Discover extractable fields with types and confidence scores
  - **Schema Generation**: Create reusable extraction templates from selected fields
  - **Data Extraction**: Extract structured data using JSON schemas
- Tools in `tools/`:
  - `base.py`: Shared utilities and base classes
  - `field_analyzer.py`: Analyzes documents to discover extractable fields
  - `schema_generator.py`: Generates JSON schemas from selected fields
  - `data_extractor.py`: Extracts structured data using schemas
- Inherits from `BaseAgent` for shared session/rate-limiting/memory
- GCS schema caching with SHA-256 validation
- Document type detection (invoice, contract, receipt, etc.)
- Line-item detection for invoice/receipt extraction

### Core Agent Infrastructure (`src/agents/core/`)

**Base Agent** (`base_agent.py`):
- Abstract `BaseAgent` class eliminating code duplication across agents
- Shared functionality:
  - Session management and rate limiting initialization
  - Memory initialization (short-term + long-term)
  - Audit logging setup with `ThreadPoolExecutor`
  - Token tracking callback factory
  - Health status reporting framework
  - Session ending and conversation summary persistence
  - Graceful resource cleanup via `shutdown()`
- Inheritors: `SheetsAgent`, `DocumentAgent`, `ExtractorAgent`

**Base Config** (`base_config.py`):
- Shared configuration inherited by all agent configs
- Common settings: `temperature`, `timeout_seconds`, `max_retries`
- Session settings: `session_timeout_minutes`
- Rate limiting: `rate_limit_requests`, `rate_limit_window_seconds`
- Memory: `enable_short_term_memory`, `enable_long_term_memory`, `short_term_max_messages`

**Rate Limiting** (`rate_limiter.py`):
- `RateLimiter` class - Thread-safe sliding window algorithm
- Tracks requests per session with configurable limits
- Methods: `is_allowed()`, `get_retry_after()`, `get_remaining()`, `cleanup()`
- Default: 10 requests per 60 seconds per session

**Session Management** (`session_manager.py`):
- `SessionInfo` model - Session data with performance tracking
  - Fields: `session_id`, `created_at`, `last_activity`, `expires_at`
  - Performance: `query_count`, `total_tokens_used`, `total_processing_time_ms`
  - Context: `documents_processed`, `files_processed`, `files_in_context`
- `SessionManager` class - Thread-safe session lifecycle
  - 30-minute timeout with activity extension
  - Per-session response caching (10 responses LRU)
  - Methods: `get_or_create_session()`, `update_session()`, `cleanup_expired_sessions()`

**Memory** (`memory/`):
- `ShortTermMemory`: In-memory conversation history per session (max 20 messages)
- `PostgresLongTermMemory`: Persistent conversation summaries
- Automatic cleanup: Conversation summaries saved on session end

**Middleware** (`middleware/`):
- `MiddlewareStack`: Composes resilience, safety, and tool selection
- `ModelRetry`, `ToolRetry`: Exponential backoff retry logic (3 attempts)
- `ModelFallback`: Primary/fallback model switching
- `PIIDetector`: Input/output PII redaction (redact/mask/hash/block strategies)
- `CallLimitTracker`: Prevents runaway tool/model calls
- `LLMToolSelector`: Pre-filters tools based on query relevance
- `QueryClassifier`: Rule-based + LLM intent detection (RAG_SEARCH, CONTENT_GENERATION, MIXED)

**Tool Selection** (`middleware/query_classifier.py`, `middleware/tool_selector.py`):
- Two-stage hybrid routing for DocumentAgent
- Stage 1: `QueryClassifier` detects intent (Q&A vs content generation)
- Stage 2: `LLMToolSelector` narrows tools within category
- RAG queries → only `rag_search` tool
- Generation queries → loader + generators + persist
- Config: `TOOL_SELECTOR_MODEL`, `ENABLE_TOOL_SELECTION`, `TOOL_SELECTOR_MAX_TOOLS`

### Usage & Quota Module (`src/core/usage/`)

Token tracking, quota enforcement, and subscription management:

- `service.py`: `UsageTrackingService` singleton
  - `log_token_usage()`: Log token consumption per feature
  - `log_resource_usage()`: Track LlamaParse pages, file search queries, storage
  - `get_usage_summary()`: Current period usage with breakdowns
  - `get_subscription()`, `create_subscription()`: Subscription CRUD
- `quota_checker.py`: `QuotaChecker` with caching for fast limit checks
- `callback_handler.py`: `TokenTrackingCallbackHandler` for LangChain integration
- `token_extractors.py`: Extract token counts from OpenAI/Gemini responses
- `decorators.py`: `@check_quota`, `@track_resource`, `@track_tokens` for endpoint protection
- `schemas.py`: `TokenUsage`, `QuotaStatus`, `UsageSummary` Pydantic models
- `context.py`: `UsageContext` context manager for request-scoped usage tracking
- `usage_queue.py`: Background queue for non-blocking usage tracking
  - `UsageQueue`: Thread-safe queue with background worker
  - `enqueue_token_usage()`, `enqueue_resource_usage()`: Non-blocking queue functions
- `models.py`: SQLAlchemy models for `subscription_tiers`, `organization_subscriptions`, `token_usage_records`

**Subscription Tiers** (Free/Pro/Enterprise):
- Token limits: 50K / 500K / 5M monthly
- LlamaParse pages: 50 / 500 / 5K monthly
- File search queries: 100 / 1K / 10K monthly
- Storage: 1 / 10 / 100 GB

### Bulk Processing Module (`src/bulk/`)

Concurrent document processing with LangGraph state machine orchestration:

- `config.py`: `BulkProcessingConfig` with 25+ settings via `BULK_` prefix env vars
  - Limits: `max_documents_per_folder` (10), `max_file_size_mb` (50)
  - Concurrency: `concurrent_documents` (3)
  - Timeouts: `parse_timeout_seconds` (300), `generation_timeout_seconds` (300), `job_timeout_seconds` (3600)
  - Retry: `max_retries_per_document` (3), `retry_delay_seconds` (30)
- `schemas.py`: Domain models for `BulkJob`, `BulkDocumentItem`, `ProcessingEvent`, `ProcessingOptions`
- `service.py`: `BulkProcessingService` main orchestration
  - Job creation with quota enforcement
  - Progress tracking with status caching (3s TTL)
  - Document retry and job cancellation
- `state_graph.py`: LangGraph `StateGraph` for workflow
  - States: parse → index → generate → finalize
  - PostgreSQL checkpointing for durability
  - Parallel document processing within concurrency limits
- `generation.py`: Content generation helpers
  - `run_generators_parallel()`: Parallel summary/FAQ/question generation
  - `parse_generation_results()`: Parse JSON results from generation tools
  - `persist_generated_content()`: Save generated content to GCS
  - `track_generation_usage()`: Non-blocking usage tracking
- `folder_manager.py`: `BulkFolderManager` for folder operations
  - Create folders, generate signed upload URLs
  - List and manage documents in folders
- `queue.py`: `ProcessingQueue` for async event handling
- `webhook_handler.py`: `WebhookHandler` for Cloud Function triggers

**Supported File Types:**
PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, CSV, TXT, RTF, HTML, images (JPG, PNG, GIF, BMP, TIFF, WebP)

### Executor Pool Infrastructure (`src/core/executors.py`)

Centralized thread pool management preventing resource starvation:

- `ExecutorRegistry` class managing three dedicated pools:
  - `agent_executor`: LLM agent invocations (10 threads via `AGENT_EXECUTOR_POOL_SIZE`)
  - `io_executor`: GCS/file I/O operations (20 threads via `IO_EXECUTOR_POOL_SIZE`)
  - `query_executor`: DuckDB/SQL queries (10 threads via `QUERY_EXECUTOR_POOL_SIZE`)
- Separates long-running LLM calls from quick I/O operations
- Graceful shutdown with optional future cancellation

### Parse Service (`src/services/parse_service.py`)

Shared document parsing abstraction:

- Unified interface for parsing PDFs, images, and documents
- GCS integration for upload/download of raw and parsed files
- LlamaParse integration with OCR and handwriting support
- Cache checking for already-parsed documents
- Database registration of parsed documents
- `ParseResult` dataclass with timing and page estimates

### Database (`src/db/`)

- `connection.py`: `DatabaseManager` singleton with per-event-loop connection management (supports background threads)
- `models.py`: SQLAlchemy ORM models (re-exports from biz2bricks_core):
  - **Core tables**: `Organization`, `User`, `Folder`, `Document`, `AuditLog`
  - **Processing tables**: `ProcessingJob`, `DocumentGeneration`
  - **Memory tables**: `UserPreference`, `ConversationSummary`, `MemoryEntry`
  - **RAG tables**: `FileSearchStore`, `DocumentFolder`
- `repositories/`:
  - `audit_repository.py`: DEPRECATED - re-exports from `audit/` subpackage for backwards compatibility
  - `audit/`: Split audit repositories (preferred imports):
    - `document_repository.py`: Document CRUD operations (`register_document`, `get_document_by_name`, etc.)
    - `job_repository.py`: Job lifecycle (`start_job`, `complete_job`, `fail_job`, `find_cached_result`)
    - `audit_log_repository.py`: Event logging (`log_event`, `get_audit_trail`, `get_document_summary`)
    - `generation_repository.py`: Generated content (`save_document_generation`, `find_cached_generation`)
    - `stats_repository.py`: Dashboard statistics (`get_dashboard_stats`, `count_jobs`, `count_documents`)
  - `bulk_repository.py` (929 lines): Bulk job persistence
    - Job CRUD: `create_bulk_job`, `get_bulk_job`, `update_job_status`
    - Document items: `add_document_item`, `update_document_status`, `get_documents_by_job`
    - Progress tracking: `get_job_progress`, `get_pending_documents`
  - `extraction_repository.py` (777 lines): Extraction data persistence
    - Jobs: `create_extraction_job`, `get_extraction_job`, `update_extraction_status`
    - Templates: `save_template`, `get_template`, `list_templates`
    - Records: `save_extracted_record`, `get_extracted_records`
  - `memory_repository.py`: Long-term memory persistence
  - `rag_repository.py`: File store and folder management
- `utils.py`: Database utility functions

Cloud SQL Connector is used in production with automatic fallback to direct connection if unavailable.

### RAG Module (`src/rag/`)

- `llama_parse_util.py`: Document parsing (PDF, DOCX, PPTX, images with OCR, handwriting support)
  - Uses Gemini-2.5-pro multimodal LLM for enhanced table/image extraction
  - Markdown output with proper formatting
- `gemini_file_store.py`: Gemini File Search API for semantic retrieval
  - `query_store()` function supports hybrid search modes (semantic/keyword/hybrid)
  - File and folder metadata filtering
  - Answer generation with citations
- `file_search_service.py`: High-level orchestration

#### Semantic Search Implementation

**Search Modes** (`src/rag/gemini_file_store.py:query_store()`):
- `semantic` - Pure vector similarity search (default)
- `keyword` - BM25 keyword matching with query enhancement
- `hybrid` - Combined semantic + keyword search

**Request Schema** (`src/api/schemas/rag.py:SearchStoreRequest`):
```python
class SearchStoreRequest(BaseModel):
    query: str
    top_k: int = 5  # 1-20
    file_filter: Optional[str] = None  # Filter by file name
    folder_name: Optional[str] = None  # Filter by folder
    folder_id: Optional[str] = None
    search_mode: str = "semantic"  # semantic|keyword|hybrid
    generate_answer: bool = True
```

**Search Scopes**:
| Scope | Filter Parameters | Description |
|-------|-------------------|-------------|
| Single File | `file_filter: "doc.pdf"` | Search one file |
| Folder | `folder_name: "Legal"` | Search all files in folder |
| Org-wide | No filters | Search ALL org documents |

**Multi-Tenancy**:
- Each org has one store: `<org_name>_file_search_store`
- `X-Organization-ID` header enforces access control
- `_validate_store_ownership()` validates store ownership

### Storage Module (`src/storage/`)

Google Cloud Storage integration for parsed documents and generated content.

- `base.py`: Abstract `StorageBackend` interface with methods: `save()`, `read()`, `exists()`, `list_files()`, `delete()`, `get_uri()`
- `gcs.py`: `GCSStorage` implementation using `google-cloud-storage` library with ADC authentication
- `config.py`: `StorageConfig` (Pydantic) and `get_storage()` singleton factory

**GCS Paths:**
- Parsed docs: `gs://{GCS_BUCKET}/{GCS_PREFIX}/{PARSED_DIRECTORY}/` (e.g., `gs://biz2bricks-dev-v1-document-store/demo_docs/parsed/`)
- Generated content: `gs://{GCS_BUCKET}/{GCS_PREFIX}/{GENERATED_DIRECTORY}/` (e.g., `gs://biz2bricks-dev-v1-document-store/demo_docs/generated/`)

### Cloud Functions (`cloud_functions/`)

GCS-triggered serverless functions for bulk processing:

**Bulk Trigger** (`bulk_trigger/main.py`):
- Cloud Function triggered by GCS object finalization events
- Monitors configured bucket for new uploads in bulk folders
- Extracts `org_id` and `folder_name` from GCS path pattern: `{org}/{bulk_prefix}/{folder}/`
- Sends webhook notifications to main application API
- Configurable via environment variables:
  - `API_BASE_URL`: Target API endpoint
  - `WEBHOOK_SECRET`: Shared authentication secret
  - `BULK_FOLDER_PREFIX`: Path prefix to monitor (default: "bulk")

### Utility Modules (`src/utils/`)

**gcs_utils.py** - GCS path parsing and building:
- `is_gcs_path(path)` - Check if path starts with `gs://`
- `parse_gcs_uri(uri)` - Returns (bucket, blob_path) tuple
- `extract_org_from_gcs_path(uri)` - Extract org name from path
- `build_gcs_uri(bucket, *parts)` - Build GCS URI from components
- `strip_gcs_prefix(uri)` - Remove `gs://` prefix
- `extract_gcs_path_parts(uri, max_splits)` - Split path into list

**timer_utils.py** - Performance timing:
- `elapsed_ms(start_time)` - Calculate elapsed milliseconds from `time.time()` start
- `Timer` - Context manager with `start()`, `stop()`, `elapsed_ms` property

**async_utils.py** - Async/sync interoperability:
- `run_async(coro)` - Run async coroutine from sync context (handles nested event loops)
- `run_sync_in_executor(func, *args)` - Run blocking sync function in thread pool

**env_utils.py** - Environment variable parsing:
- `parse_bool_env(key, default)` - Parse boolean from env var (handles "true"/"false")
- `parse_int_env(key, default)` - Parse integer from env var
- `parse_float_env(key, default)` - Parse float from env var

### Constants (`src/constants.py`)

Centralized application constants replacing magic numbers throughout codebase:

```python
# GCS
GCS_URI_PREFIX = "gs://"
GCS_URI_PREFIX_LEN = 5

# Caching
DEFAULT_FILE_CACHE_SIZE = 50
DEFAULT_RESPONSE_CACHE_SIZE = 10
DEFAULT_PREVIEW_LENGTH = 500

# Session & Rate Limiting
DEFAULT_SESSION_TIMEOUT_MINUTES = 30
DEFAULT_RATE_LIMIT_REQUESTS = 10
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60

# Agent Timeouts
DEFAULT_AGENT_TIMEOUT_SECONDS = 300
DEFAULT_CLEANUP_INTERVAL_SECONDS = 300

# Database
DEFAULT_DB_POOL_SIZE = 5
DEFAULT_DB_MAX_OVERFLOW = 10

# Content Generation
DEFAULT_NUM_FAQS = 10
DEFAULT_NUM_QUESTIONS = 10
DEFAULT_SUMMARY_MAX_WORDS = 500

# Validation Bounds (added in refactoring)
MIN_NUM_FAQS = 1
MAX_NUM_FAQS = 50
MIN_NUM_QUESTIONS = 1
MAX_NUM_QUESTIONS = 100
MIN_SUMMARY_WORDS = 50
MAX_SUMMARY_WORDS = 2000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

# Event Types (for audit logging)
EVENT_GENERATION_CACHE_HIT = "generation_cache_hit"
EVENT_GENERATION_STARTED = "generation_started"
EVENT_GENERATION_COMPLETED = "generation_completed"

# Search Modes
VALID_SEARCH_MODES = ["semantic", "keyword", "hybrid"]
```

## Key Patterns

- **Multi-Tenancy**: Organization context from `X-Organization-ID` header, all operations scoped by org_id
- **Session Management**: 30-minute timeout, per-session response caching (10 cached responses per session)
- **Rate Limiting**: 10 requests per 60-second window per session via `RateLimiter` class
- **Caching**: LRU file cache (50 files via `FileCache`), response cache via query hash, GCS cache with SHA-256 hash, PostgreSQL generation cache
- **Retry Logic**: Exponential backoff with tenacity (3 attempts), configurable in middleware
- **Security**: Path traversal prevention, SQL injection blocking (`DANGEROUS_SQL_KEYWORDS`), PII detection
- **Async Background Tasks**: `ThreadPoolExecutor` for audit logging, periodic cleanup task for expired sessions

## API Endpoints

### Health
- `GET /health` - Service health status

### Documents (`/api/v1/documents/`)
- `POST /process` - Process with flexible query
- `POST /summarize` - Generate summary (with GCS cache check)
- `POST /faqs` - Generate FAQs (with GCS cache check)
- `POST /questions` - Generate comprehension questions (with GCS cache check)
- `POST /generate-all` - Generate all content types (with GCS cache check)
- `POST /chat` - Conversational RAG with citations (uses `rag_search` tool)

### Sheets (`/api/v1/sheets/`)
- `POST /analyze` - Natural language query on Excel/CSV
- `POST /preview` - Preview file contents and metadata
- `GET /health` - Sheets agent health status

### Ingestion (`/api/v1/ingest/`)
- `POST /upload` - Upload files (org-scoped storage)
- `POST /parse` - Parse documents via LlamaParse
- `GET /files` - List uploaded files
- `GET /files/{file_id}` - Get file details
- `DELETE /files/{file_id}` - Delete file

### RAG (`/api/v1/rag/`)
- `POST /stores` - Create file search store
- `GET /stores` - List stores
- `GET /stores/{store_id}` - Get store details
- `POST /stores/{store_id}/upload` - Upload document to store
- `POST /stores/{store_id}/search` - Semantic search
- `GET /stores/{store_id}/files` - List files in store
- `DELETE /stores/{store_id}` - Delete store
- `POST /folders` - Create folder
- `GET /folders` - List folders
- `GET /folders/{folder_id}` - Get folder details
- `DELETE /folders/{folder_id}` - Delete folder

### Audit (`/api/v1/audit/`)
- `GET /dashboard` - Dashboard statistics with period filter (7d/30d/90d/all)
- `GET /activity` - Activity timeline with recent events
- `GET /jobs` - List processing jobs (paginated)
- `GET /jobs/{job_id}` - Get job details
- `GET /documents` - List processed documents
- `GET /documents/{doc_id}` - Get document record
- `GET /generations` - List generated content
- `GET /trail` - Audit trail with filtering

### Bulk Processing (`/api/v1/bulk/`)
- `POST /upload` - Upload multiple files and start bulk processing
- `POST /folders` - Create bulk folder
- `GET /folders` - List org folders (paginated)
- `GET /folders/{folder_id}` - Get folder details
- `DELETE /folders/{folder_id}` - Delete folder
- `POST /folders/{folder_id}/upload-urls` - Generate signed upload URLs (legacy)
- `GET /jobs` - List bulk jobs (paginated)
- `GET /jobs/{job_id}` - Get job details with progress
- `POST /jobs/{job_id}/cancel` - Cancel bulk job (idempotent)
- `POST /jobs/{document_id}/retry` - Retry failed document
- `POST /webhook/document-uploaded` - Webhook for Cloud Function triggers

### Extraction (`/api/v1/extraction/`)
Router refactored into package (`src/api/routers/extraction/`) with submodules:
- `POST /analyze-fields` - Discover extractable fields with types and confidence scores
- `POST /generate-schema` - Generate extraction template from selected fields
- `POST /extract` - Extract structured data using schema
- `GET /templates` - List extraction templates
- `GET /templates/{template_name}` - Get template details
- `GET /records` - List extracted records
- `GET /records/{record_id}` - Get extracted record details
- `POST /export` - Export extracted data (CSV, JSON)
- `GET /health` - Extraction service health status

### Sessions (`/api/v1/sessions/`)
- `GET /{session_id}` - Get session info
- `DELETE /{session_id}` - End session and cleanup

### Tiers (`/api/v1/tiers/`)
- `GET /` - List subscription tiers (public endpoint, no auth required)

### Content (`/api/v1/content/`)
- `POST /load-parsed` - Load pre-parsed document content from GCS
- `GET /check-exists` - Check if parsed document exists in GCS

### Usage (`/api/v1/usage/`)
- `GET /summary` - Current period usage summary (tokens, pages, queries, storage)
- `GET /subscription` - Subscription details and limits
- `GET /limits` - Quota status with approaching/exceeded warnings
- `GET /history` - Historical usage data (7d, 30d, 90d periods)
- `GET /breakdown` - Usage breakdown by feature/model

## Agent Configuration

All agents use Pydantic configs with environment variable defaults, inheriting from `BaseAgentConfig`:

**SheetsAgentConfig** (`src/agents/sheets/config.py`):
- `openai_model`: gpt-5.1-codex-mini (via `OPENAI_SHEET_MODEL`)
- `timeout_seconds`: 300s, `session_timeout_minutes`: 30
- `rate_limit_requests`: 10, `rate_limit_window_seconds`: 60
- `duckdb_pool_size`: 5, `max_result_rows`: 10,000
- `enable_short_term_memory`, `enable_long_term_memory`: true

**DocumentAgentConfig** (`src/agents/document/config.py`):
- `openai_model`: gpt-5-mini (via `DOCUMENT_AGENT_MODEL`)
- `default_num_faqs`: 10, `default_num_questions`: 10, `summary_max_words`: 500
- `timeout_seconds`: 300s, `session_timeout_minutes`: 30
- `rate_limit_requests`: 10, `rate_limit_window_seconds`: 60
- `persist_to_database`: true, `middleware`: MiddlewareConfig
- `enable_short_term_memory`, `enable_long_term_memory`: true
- `enable_tool_selection`: true, `tool_selector_model`: gpt-5-mini (via `TOOL_SELECTOR_MODEL`)
- `tool_selector_max_tools`: 3 (max tools per query after filtering)

**ExtractorAgentConfig** (`src/agents/extractor/config.py`):
- `openai_model`: gpt-5-mini (via `EXTRACTOR_AGENT_MODEL`)
- `openai_fallback_model`: gpt-5.2-2025-12-11 (via `EXTRACTOR_FALLBACK_MODEL`)
- `temperature`: 0.2 (lower for deterministic extraction)
- `max_fields_to_analyze`: 50, `max_schema_fields`: 100
- `extraction_timeout_seconds`: 120
- `schemas_directory`: schemas, `extracted_directory`: extracted
- `persist_to_database`: true

## Usage Examples

```python
# SheetsAgent
from src.agents.sheets.core import SheetsAgent
from src.agents.sheets.schemas import ChatRequest
from src.agents.sheets.config import SheetsAgentConfig

agent = SheetsAgent(SheetsAgentConfig())
response = await agent.process_chat(ChatRequest(
    file_paths=["file.xlsx"],
    query="What's the total amount?"
))
# Clean shutdown
agent.shutdown(wait=True)

# DocumentAgent
from src.agents.document.core import DocumentAgent
from src.agents.document.config import DocumentAgentConfig

agent = DocumentAgent()  # Uses default config
# Or with custom config:
# agent = DocumentAgent(DocumentAgentConfig(openai_model="gpt-5-nano"))

summary = await agent.generate_summary("Sample1.md")
faqs = await agent.generate_faqs("Sample1.md", num_faqs=5)
questions = await agent.generate_questions("Sample1.md", num_questions=10)
content = await agent.generate_all("Sample1.md")

# Clean shutdown - waits for pending audit logs
agent.shutdown(wait=True)
```

## Directory Conventions

**GCS Storage:**
- `gs://{bucket}/{prefix}/parsed/`: Pre-parsed documents in Markdown format (from LlamaParse)
- `gs://{bucket}/{prefix}/generated/`: Generated content output (JSON files with summaries, FAQs, questions)

**Local Storage:**
- `/upload/`: Raw user uploads (text files only, PDFs require pre-parsing)
- `/docs/`: Sample documents for testing
- `/parsed/`: Local parsed documents (alternative to GCS for development)

**Scripts:**
- `scripts/db_setup.py`: Database setup, teardown, reset, status, and SQL export
- `scripts/seed_tiers.py`: Seed subscription tiers (Free/Pro/Enterprise)
- `scripts/migrate_orgs_free.py`: Migrate organizations to free tier
- `scripts/migrate_documents_status.py`: Data migration utility

**Tests:**
- `tests/unit/`: Unit tests for agents, API routers, and storage
- `tests/integration/`: Integration tests for GCS and end-to-end flows
- `tests/conftest.py`: Shared pytest fixtures

## Key Implementation Files

| Feature | File | Description |
|---------|------|-------------|
| App Factory | `src/api/app.py` | Lifespan management, periodic cleanup |
| Token Tracking | `src/api/usage.py` | Usage tracking and cost estimation |
| Error Schemas | `src/api/schemas/errors.py` | Shared error response definitions |
| Shared Validators | `src/api/schemas/validators.py` | Path, query, options validation |
| Constants | `src/constants.py` | Centralized application constants |
| Env Utilities | `src/utils/env_utils.py` | Boolean/int/float env parsing |
| GCS Utilities | `src/utils/gcs_utils.py` | Path parsing and URI building |
| Timer Utilities | `src/utils/timer_utils.py` | Performance timing helpers |
| Async Utilities | `src/utils/async_utils.py` | Async/sync interoperability |
| Base Agent | `src/agents/core/base_agent.py` | Abstract base for all agents |
| Base Config | `src/agents/core/base_config.py` | Shared agent configuration |
| Rate Limiting | `src/agents/core/rate_limiter.py` | Per-session sliding window limits |
| Session Mgmt | `src/agents/core/session_manager.py` | Session lifecycle and caching |
| GCS Caching | `src/agents/document/gcs_cache.py` | SHA-256 based content caching |
| Tool Selection | `src/agents/document/tool_selection.py` | ToolSelectionManager + filters |
| Result Parser | `src/agents/document/result_parser.py` | Agent output parsing |
| Tool Factory | `src/agents/document/tools/__init__.py` | Document tools creation |
| Extractor Agent | `src/agents/extractor/core.py` | Structured data extraction |
| File Cache | `src/agents/sheets/cache.py` | LRU DataFrame caching |
| DuckDB Pool | `src/agents/sheets/tools.py` | Connection pooling for SQL queries |
| Middleware Stack | `src/agents/core/middleware/stack.py` | Composable middleware |
| Executor Pools | `src/core/executors.py` | Thread pool registry |
| Parse Service | `src/services/parse_service.py` | Shared document parsing |
| Bulk Service | `src/bulk/service.py` | Bulk job orchestration |
| Bulk Graph | `src/bulk/state_graph.py` | LangGraph workflow |
| DB Connection | `src/db/connection.py` | Per-event-loop async sessions |
| Audit Repositories | `src/db/repositories/audit/` | Split document/job/log/generation repos |
| Bulk Repository | `src/db/repositories/bulk_repository.py` | Bulk job persistence |
| Extraction Repository | `src/db/repositories/extraction_repository.py` | Extraction data persistence |
| RAG Repository | `src/db/repositories/rag_repository.py` | Store/folder management |
| Stats Repository | `src/db/repositories/audit/stats_repository.py` | Dashboard statistics |
| Usage Service | `src/core/usage/service.py` | Token/resource tracking |
| Quota Checker | `src/core/usage/quota_checker.py` | Limit enforcement with caching |
| Usage Queue | `src/core/usage/usage_queue.py` | Background usage tracking |
| Bulk Router | `src/api/routers/bulk.py` | Bulk processing endpoints |
| Extraction Router | `src/api/routers/extraction/` | Extraction API endpoints (package) |
| Audit Router | `src/api/routers/audit.py` | Dashboard and audit endpoints |
| Tiers Router | `src/api/routers/tiers.py` | Subscription tiers (public) |
| Content Router | `src/api/routers/content.py` | Content loading from GCS |
| API Utils | `src/api/utils/` | Formatting, decorators, responses |
| Usage Router | `src/api/routers/usage.py` | Usage/subscription endpoints |
| Bulk Generation | `src/bulk/generation.py` | Content generation helpers |
