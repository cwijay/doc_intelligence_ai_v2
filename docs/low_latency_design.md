# Low-Latency Design - Document Intelligence AI v3.0

This document describes the async programming patterns and low-latency design choices implemented in the Document Intelligence AI system to support thousands of concurrent users.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Executor Pool Design](#2-executor-pool-design)
3. [Async I/O Patterns](#3-async-io-patterns)
4. [Caching Strategy](#4-caching-strategy)
5. [Connection Pooling](#5-connection-pooling)
6. [Background Queues](#6-background-queues)
7. [Rate Limiting](#7-rate-limiting)
8. [Connection Pre-warming](#8-connection-pre-warming)
9. [Lifecycle Management](#9-lifecycle-management)
10. [Performance Impact Summary](#10-performance-impact-summary)

---

## 1. System Architecture Overview

### High-Level Request Flow

```
                                    +------------------+
                                    |   Load Balancer  |
                                    +--------+---------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                              FASTAPI APPLICATION                                   |
|                                                                                   |
|  +-------------+    +------------------+    +------------------+                  |
|  |   Uvicorn   |--->|  Event Loop      |--->|  Rate Limiter    |                  |
|  | (ASGI)      |    |  (asyncio)       |    |  (per-session)   |                  |
|  +-------------+    +--------+---------+    +--------+---------+                  |
|                              |                       |                            |
|                              v                       v                            |
|                     +--------+---------+    +--------+---------+                  |
|                     |  Session Manager |    |  Quota Checker   |                  |
|                     |  (LRU cache)     |    |  (TTL cache)     |                  |
|                     +--------+---------+    +--------+---------+                  |
|                              |                                                    |
+------------------------------|----------------------------------------------------+
                               |
          +--------------------+--------------------+
          |                    |                    |
          v                    v                    v
   +------+------+      +------+------+      +------+------+
   | Agent Pool  |      |  I/O Pool   |      | Query Pool  |
   | (10 threads)|      | (20 threads)|      | (10 threads)|
   +------+------+      +------+------+      +------+------+
          |                    |                    |
          v                    v                    v
   +------+------+      +------+------+      +------+------+
   |  LLM Calls  |      |  GCS I/O    |      |  DuckDB     |
   | (5-30 sec)  |      | (100-500ms) |      | (1-5 sec)   |
   +-------------+      +-------------+      +-------------+

+-----------------------------------------------------------------------------------+
|                           BACKGROUND WORKERS                                       |
|                                                                                   |
|  +----------------+    +----------------+    +--------------------+               |
|  |  Audit Queue   |    |  Usage Queue   |    |  Bulk Job Queue    |               |
|  | (concurrent:1) |    | (concurrent:1) |    | (concurrent:N)     |               |
|  +-------+--------+    +-------+--------+    +---------+----------+               |
|          |                     |                       |                          |
|          v                     v                       v                          |
|  +----------------+    +----------------+    +--------------------+               |
|  |  PostgreSQL    |    |  PostgreSQL    |    |    PostgreSQL      |               |
|  | (pool_size:2)  |    | (pool_size:2)  |    |   (pool_size:2)    |               |
|  +----------------+    +----------------+    +--------------------+               |
+-----------------------------------------------------------------------------------+
```

### Design Principles

1. **Non-Blocking Event Loop**: Never block the main asyncio event loop
2. **Workload Isolation**: Separate thread pools for different operation types
3. **Multi-Layer Caching**: Cache at every layer to reduce latency
4. **Background Processing**: Defer non-critical operations (audit, usage logging)
5. **Graceful Degradation**: Fallbacks when services are unavailable

---

## 2. Executor Pool Design

**File:** `src/core/executors.py`

### Architecture

```
+------------------------------------------------------------------+
|                      ExecutorRegistry                             |
|                         (Singleton)                               |
+------------------------------------------------------------------+
|                                                                  |
|  +-------------------+  +-------------------+  +----------------+ |
|  |   Agent Executor  |  |    I/O Executor   |  | Query Executor | |
|  |                   |  |                   |  |                | |
|  | max_workers: 10   |  | max_workers: 20   |  | max_workers: 10| |
|  | prefix: "agent-"  |  | prefix: "io-"     |  | prefix: "query"| |
|  |                   |  |                   |  |                | |
|  | Used for:         |  | Used for:         |  | Used for:      | |
|  | - LLM invocations |  | - GCS read/write  |  | - DuckDB SQL   | |
|  | - Agent execution |  | - File system I/O |  | - Data queries | |
|  | - 5-30 sec ops    |  | - 100-500ms ops   |  | - 1-5 sec ops  | |
|  +-------------------+  +-------------------+  +----------------+ |
+------------------------------------------------------------------+
```

### Configuration

| Executor | Default Size | Environment Variable | Use Case |
|----------|--------------|---------------------|----------|
| Agent    | 10 threads   | `AGENT_EXECUTOR_POOL_SIZE` | Heavy LLM operations |
| I/O      | 20 threads   | `IO_EXECUTOR_POOL_SIZE` | GCS, filesystem |
| Query    | 10 threads   | `QUERY_EXECUTOR_POOL_SIZE` | DuckDB, SQL |

### Monitoring

The ExecutorRegistry provides a `get_stats()` method for observability:

```python
from src.core.executors import get_executors

stats = get_executors().get_stats()
# Returns:
# {
#     "agent_pool": {"max_workers": 10, "thread_prefix": "agent-"},
#     "io_pool": {"max_workers": 20, "thread_prefix": "io-"},
#     "query_pool": {"max_workers": 10, "thread_prefix": "query-"}
# }
```

### Why Separate Pools?

```
WITHOUT SEPARATION:                    WITH SEPARATION:
+------------------+                   +--------+ +--------+ +--------+
| Single Pool (30) |                   | Agent  | |  I/O   | | Query  |
+------------------+                   | (10)   | | (20)   | | (10)   |
         |                             +--------+ +--------+ +--------+
         v                                  |          |          |
  [LLM 30s][LLM 30s]...                    |          |          |
  [GCS 0.1s] BLOCKED!                      v          v          v
  [Query 2s] BLOCKED!                   [LLM]      [GCS]      [Query]
                                        [LLM]      [GCS]      [Query]
                                          |          |          |
Result: I/O waits behind LLMs          I/O runs in parallel with LLMs
```

---

## 3. Async I/O Patterns

### Pattern 1: run_in_executor for Blocking I/O

**File:** `src/storage/gcs.py`

```python
async def save(self, content: str, filename: str) -> str:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        get_executors().io_executor,  # Dedicated I/O pool
        partial(blob.upload_from_string, content),
    )
```

**Request Flow:**

```
Main Event Loop              I/O Executor Pool
     |                              |
     | await run_in_executor()      |
     |----------------------------->|
     |                              | blob.upload() [BLOCKING]
     | (continues other requests)   |
     |                              | ...done
     |<-----------------------------|
     | result                       |
```

### Pattern 2: asyncio.gather() for Parallel Operations

**File:** `src/api/routers/documents.py`

```python
# Sequential (slow - 300ms total):
summary = await check_cached_summary()   # 100ms
faqs = await check_cached_faqs()         # 100ms
questions = await check_cached_questions() # 100ms

# Parallel (fast - 100ms total):
summary, faqs, questions = await asyncio.gather(
    check_cached_summary(),
    check_cached_faqs(),
    check_cached_questions()
)
```

**Timing Diagram:**

```
SEQUENTIAL:
|--summary (100ms)--|--faqs (100ms)--|--questions (100ms)--|
                                                           Total: 300ms

PARALLEL:
|--summary (100ms)--|
|--faqs (100ms)-----|
|--questions (100ms)-|
                     Total: 100ms (3x faster)
```

### Pattern 3: Async Context Manager for Resources

**File:** `src/db/connection.py`

```python
@asynccontextmanager
async def session() -> AsyncGenerator[AsyncSession, None]:
    session = self._session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
```

---

## 4. Caching Strategy

### Multi-Layer Cache Hierarchy

```
+------------------------------------------------------------------+
|                         REQUEST                                   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  LAYER 1: Session Response Cache (in-memory)                     |
|  - Per-session LRU cache (10 responses)                          |
|  - Key: query_hash                                                |
|  - Hit: ~0ms (instant return)                                     |
+------------------------------------------------------------------+
                              | miss
                              v
+------------------------------------------------------------------+
|  LAYER 2: Quota/Subscription Cache (in-memory)                   |
|  - TTL: 60 seconds                                                |
|  - Async lock for concurrent access                               |
|  - Hit: ~0ms (avoids DB lookup)                                   |
+------------------------------------------------------------------+
                              | miss
                              v
+------------------------------------------------------------------+
|  LAYER 3: File Cache / DataFrame Cache (in-memory)               |
|  - LRU with OrderedDict (50 files)                                |
|  - Key: file_path                                                 |
|  - Hit: ~1ms (avoids file re-read)                                |
+------------------------------------------------------------------+
                              | miss
                              v
+------------------------------------------------------------------+
|  LAYER 4: GCS Content Cache (distributed)                        |
|  - SHA-256 hash validation                                        |
|  - Persists summaries, FAQs, questions                            |
|  - Hit: ~50-100ms (avoids LLM call)                               |
+------------------------------------------------------------------+
                              | miss
                              v
+------------------------------------------------------------------+
|  LAYER 5: Database (PostgreSQL)                                   |
|  - Connection pooled (5 + 10 overflow)                            |
|  - Async with asyncpg                                             |
|  - ~10-50ms per query                                             |
+------------------------------------------------------------------+
                              | miss
                              v
+------------------------------------------------------------------+
|  LAYER 6: LLM / External API                                      |
|  - OpenAI / Gemini                                                |
|  - 1-30 seconds per call                                          |
+------------------------------------------------------------------+
```

### Cache Implementations

| Layer | Type | TTL | Size | File |
|-------|------|-----|------|------|
| Session Response | LRU | Session lifetime | 10/session | `session_manager.py` |
| Quota | TTL | 60s | Unlimited | `quota_checker.py` |
| File/DataFrame | LRU | Unlimited | 50 files | `cache.py` |
| GCS Content | Persistent | Unlimited | GCS storage | `gcs_cache.py` |
| Tier Config | TTL | 1 hour | 10 tiers | `subscription_manager.py` |
| Store Metadata | TTL | 5 minutes | Unlimited | `gemini_file_store.py` |

### File Cache Statistics

**File:** `src/agents/sheets/cache.py`

The `FileCache` class tracks hit/miss statistics for observability:

```python
from src.agents.sheets.cache import FileCache

cache = FileCache(max_size=50)

# After some usage...
stats = cache.get_stats()
# Returns:
# {
#     "size": 35,              # Current cached files
#     "max_size": 50,          # Maximum capacity
#     "hits": 142,             # Cache hits
#     "misses": 28,            # Cache misses
#     "hit_rate_percent": 83.5 # Hit rate percentage
# }
```

---

## 5. Connection Pooling

### SQLAlchemy Async Pool

**File:** `src/db/connection.py`

The DatabaseManager implements per-event-loop engine management to avoid "Future attached to different loop" errors when using background threads.

**Key Features:**
- Singleton pattern with per-loop resource tracking
- Differentiated pool sizes for main vs background loops
- Cloud SQL Connector with automatic fallback to direct connection
- 30-second connection timeout with graceful fallback

```
+------------------------------------------------------------------+
|                    DatabaseManager                                |
|                      (Singleton)                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Per-Event-Loop Resource Tracking:                               |
|  _connectors: Dict[loop_id, Connector]      # Cloud SQL          |
|  _engines: Dict[loop_id, AsyncEngine]                            |
|  _session_factories: Dict[loop_id, async_sessionmaker]           |
|  _main_loop_id: int                          # First loop seen   |
|                                                                  |
+------------------------------------------------------------------+

Pool Size Differentiation:
+------------------------------------------------------------------+
|                                                                  |
|  MAIN LOOP (API requests):        BACKGROUND LOOPS (queues):     |
|  +------------------------+       +------------------------+      |
|  |   AsyncAdaptedQueuePool|       |   AsyncAdaptedQueuePool|      |
|  |   - pool_size: 3       |       |   - pool_size: 2       |      |
|  |   - max_overflow: 5    |       |   - max_overflow: 5    |      |
|  |   - pool_timeout: 30s  |       |   - pool_timeout: 30s  |      |
|  |   - pool_recycle: 30m  |       |   - pool_recycle: 30m  |      |
|  +------------------------+       +------------------------+      |
|              |                               |                    |
|              v                               v                    |
|  +-----+-----+-----+             +-----+-----+                    |
|  |C1   |C2   |C3   |             |C1   |C2   |                    |
|  +-----+-----+-----+             +-----+-----+                    |
|              +                         +                          |
|  +-----+-----+-----+-----+-----+  +-----+-----+-----+-----+-----+ |
|  | Overflow (up to 5)        |  | Overflow (up to 5)        |   |
|  +---------------------------+  +---------------------------+   |
|                                                                  |
+------------------------------------------------------------------+
```

### Pool Statistics

The DatabaseManager provides `get_pool_stats()` for monitoring:

```python
from src.db.connection import db

stats = db.get_pool_stats()
# Returns:
# {
#     "pools_count": 3,  # Number of event loops with engines
#     "shutdown_mode": False,
#     "pools": {
#         "140234567890": {"size": 3, "checked_out": 1, "overflow": 0, "checked_in": 2},
#         "140234567891": {"size": 2, "checked_out": 0, "overflow": 0, "checked_in": 2},
#     }
# }
```

### DuckDB Connection Pool

**File:** `src/agents/sheets/tools.py`

```
+------------------------------------------------------------------+
|                      DuckDBPool                                   |
|                   (Thread-Safe LRU)                               |
+------------------------------------------------------------------+
|                                                                  |
|  _pool: List[DuckDBPyConnection]                                 |
|  _max: 5 connections                                             |
|  _lock: threading.Lock()                                         |
|                                                                  |
|  get_connection():                                                |
|    +------------------+                                           |
|    | with self._lock: |                                           |
|    |   if pool:       |---> return pool.pop()                     |
|    |   else:          |---> return duckdb.connect(':memory:')     |
|    +------------------+                                           |
|                                                                  |
|  return_connection(conn):                                         |
|    +------------------+                                           |
|    | with self._lock: |                                           |
|    |   if len < max:  |---> pool.append(conn)                     |
|    |   else:          |---> conn.close()                          |
|    +------------------+                                           |
+------------------------------------------------------------------+
```

---

## 6. Background Queues

### Base Queue Architecture

**File:** `src/core/queues/base_queue.py`

All background queues extend the `BackgroundQueue[T]` abstract base class, which provides:
- Thread-safe singleton pattern
- Dedicated background thread with persistent event loop
- Configurable concurrent event processing (via semaphore)
- Database connection lifecycle management
- Graceful shutdown with pending task completion

```
+------------------------------------------------------------------+
|                     MAIN EVENT LOOP                               |
|                    (API Requests)                                 |
+------------------------------------------------------------------+
       |                        |                        |
       | enqueue()              | enqueue()              | enqueue()
       | (non-blocking)         | (non-blocking)         | (non-blocking)
       v                        v                        v
+----------------+      +----------------+      +------------------+
| Audit Queue    |      | Usage Queue    |      | Bulk Job Queue   |
| (max: 1000)    |      | (max: 1000)    |      | (max: 1000)      |
+-------+--------+      +-------+--------+      +--------+---------+
        |                       |                        |
        | (thread boundary)     | (thread boundary)      | (thread boundary)
        v                       v                        v
+----------------+      +----------------+      +------------------+
| Audit Thread   |      | Usage Thread   |      | Bulk Thread      |
| concurrent: 1  |      | concurrent: 1  |      | concurrent: N    |
| - Own loop     |      | - Own loop     |      | - Own loop       |
| - Own DB pool  |      | - Own DB pool  |      | - Own DB pool    |
+-------+--------+      +-------+--------+      +--------+---------+
        |                       |                        |
        v                       v                        v
+----------------+      +----------------+      +------------------+
|  PostgreSQL    |      |  PostgreSQL    |      |   PostgreSQL     |
| (pool_size: 2) |      | (pool_size: 2) |      |  (pool_size: 2)  |
+----------------+      +----------------+      +------------------+
```

### Queue Types

| Queue | File | Max Concurrent | Events |
|-------|------|----------------|--------|
| Audit | `src/agents/core/audit_queue.py` | 1 (sequential) | generation_save |
| Usage | `src/core/usage/usage_queue.py` | 1 (sequential) | token, resource |
| Bulk Job | `src/bulk/queue.py` | Configurable | start, process_document, complete, cancel |

### Concurrent Event Processing

Queues support configurable parallelism via `_get_max_concurrent()`:

```python
class BulkJobQueue(BackgroundQueue[BulkJobEvent]):
    def _get_max_concurrent(self) -> int:
        """Return max concurrent document processing tasks."""
        from .config import get_bulk_config
        return get_bulk_config().concurrent_documents  # e.g., 5
```

```
Semaphore-Controlled Processing:

Event Queue                           Worker Thread
    |                                      |
[E1][E2][E3][E4][E5]...                   |
    |                                      |
    +------------------------------------->| semaphore = Semaphore(N)
                                           |
                          +----------------+----------------+
                          |                |                |
                          v                v                v
                    [Task E1]        [Task E2]        [Task E3]
                     (async)          (async)          (async)
                          |                |                |
                          +----------------+----------------+
                                           |
                                           v
                               [All tasks complete]
```

### Event Flow

```
API Request Thread                   Background Worker Thread
       |                                      |
       | BulkJobEvent(                        |
       |   action="process_document",         |
       |   job_id="abc123",                   |
       |   document_id="doc456"               |
       | )                                    |
       |                                      |
       | queue.enqueue(event)  -------------->| queue.get(timeout=0.5)
       |                                      |
       | return response   (non-blocking)     | async with semaphore:
       |                                      |     await _process_event(event)
       v                                      v
  [Response sent]                      [Event persisted]
  (0ms overhead)                       (async in background)
```

### Database Connection Lifecycle

Each queue manages its own database connection:

```python
async def _process_events(self) -> None:
    # Initialize DB on loop startup (once)
    await self._init_db_for_loop()

    while not shutdown:
        event = await get_event()
        try:
            await self._process_event(event)
        except ConnectionError:
            # Auto-reconnect on connection loss
            await self._reinit_db_connection()
            await self._process_event(event)  # Retry once

    # Cleanup on shutdown
    await self._cleanup_db()
```

### Graceful Shutdown

```
Shutdown Signal
       |
       v
+---------------------+
| Set shutdown_event  |---> threading.Event.set()
+---------------------+
       |
       v
+---------------------+
| Send poison pill    |---> queue.put(None)
+---------------------+
       |
       v
+---------------------+
| Wait pending tasks  |---> await asyncio.gather(*pending_tasks)
+---------------------+
       |
       v
+---------------------+
| Close DB connection |---> await _cleanup_db()
+---------------------+
       |
       v
+---------------------+
| Join thread         |---> thread.join(timeout=5.0)
+---------------------+
```

---

## 7. Rate Limiting

### Sliding Window Algorithm

**File:** `src/agents/core/rate_limiter.py`

```
+------------------------------------------------------------------+
|                     RateLimiter                                   |
|                   (Per-Session)                                   |
+------------------------------------------------------------------+
|                                                                  |
|  Configuration:                                                   |
|  - max_requests: 10                                              |
|  - window_seconds: 60                                            |
|                                                                  |
|  requests: Dict[session_id, List[timestamp]]                     |
|                                                                  |
+------------------------------------------------------------------+

Timeline for session "abc123":
|-------------------------------------------------------------> time
|
| window_start = now - 60s
| |<------------------ 60 second window ------------------>|
| |                                                        |
| v     v  v    v      v   v       v    v                  v (now)
| [t1] [t2][t3][t4]   [t5][t6]    [t7] [t8]                [t9]
| |____|                                                   |
|   ^                                                      |
|   expired (removed)            <-- 8 requests in window -->
|
| is_allowed() returns: True (8 < 10)
| After call: 9 requests in window
```

### Implementation

```python
def is_allowed(self, session_id: str) -> bool:
    with self._lock:
        now = time.time()
        window_start = now - self.window_seconds

        # Remove expired requests
        self.requests[session_id] = [
            t for t in self.requests[session_id]
            if t > window_start
        ]

        # Check limit
        if len(self.requests[session_id]) >= self.max_requests:
            return False

        # Record request
        self.requests[session_id].append(now)
        return True
```

---

## 8. Connection Pre-warming

**File:** `src/api/app.py`

Cloud SQL connector initialization takes ~1.2 seconds for the first connection. Pre-warming during startup eliminates this latency for the first real request.

### Implementation

```python
async def _prewarm_executor_db_connections():
    """Pre-warm database connections for executor thread pools."""
    executors = get_executors()
    loop = asyncio.get_running_loop()

    def run_init():
        """Sync wrapper for async init in executor thread."""
        thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(thread_loop)
        try:
            thread_loop.run_until_complete(db.get_engine_async())
        finally:
            thread_loop.close()

    # Warm up multiple executor threads (default: 3)
    num_threads = min(3, executors.agent_executor._max_workers)
    futures = [
        loop.run_in_executor(executors.agent_executor, run_init)
        for _ in range(num_threads)
    ]

    # Wait with 30-second timeout (non-blocking if startup is slow)
    await asyncio.wait_for(
        asyncio.gather(*futures, return_exceptions=True),
        timeout=30.0
    )
```

### Impact

| Scenario | First Request Latency |
|----------|----------------------|
| Without pre-warming | +1.2s (Cloud SQL connector init) |
| With pre-warming | ~0ms (already initialized) |

---

## 9. Lifecycle Management

### Startup Sequence

```
+------------------------------------------------------------------+
|                     APPLICATION STARTUP                           |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 1. Initialize Database Connection Pool                            |
|    - Create engine for main event loop                           |
|    - Session factory initialization                               |
|    - Cloud SQL Connector or direct connection                     |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 2. Start Background Queues                                        |
|    - Audit queue (dedicated thread + event loop)                 |
|    - Usage queue (dedicated thread + event loop)                 |
|    - Bulk job queue (dedicated thread + event loop)              |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 3. Pre-warm Executor DB Connections                               |
|    - Initialize DB in 3 agent executor threads                   |
|    - Eliminates ~1.2s latency on first background task           |
|    - 30s timeout (non-blocking if slow)                          |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 4. Initialize Agents                                              |
|    - Document Agent (LLM clients, tools, middleware)             |
|    - Sheets Agent (DuckDB pool, file cache)                      |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 5. Start Periodic Cleanup Task                                    |
|    - asyncio.create_task(_periodic_cleanup())                    |
|    - Runs every 5 minutes (CLEANUP_INTERVAL_SECONDS)             |
|    - Cleans expired sessions and rate limiter entries            |
+------------------------------------------------------------------+
                              |
                              v
               +------------------+
               | 6. Ready to Serve|
               +------------------+
```

### Shutdown Sequence

**Order matters:** Agents may enqueue final audit/usage logs, so queues must shut down after agents.

```
+------------------------------------------------------------------+
|                    APPLICATION SHUTDOWN                           |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 0. Cancel Periodic Cleanup Task                                   |
|    - _cleanup_task.cancel()                                       |
|    - await _cleanup_task (catch CancelledError)                  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 1. Shutdown Agents                                                |
|    - Document Agent (may enqueue final audit logs)               |
|    - Sheets Agent (clear caches, close DuckDB)                   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 2. Shutdown Executor Pools                                        |
|    - shutdown_executors(wait=True)                               |
|    - Wait for all pending futures to complete                    |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 3. Shutdown Usage Queue                                           |
|    - shutdown(wait=True, timeout=10.0)                           |
|    - Wait for pending token usage records                        |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 4. Shutdown Bulk Job Queue                                        |
|    - stop_bulk_queue(wait=True)                                  |
|    - Wait for pending bulk processing jobs                       |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 5. Shutdown Audit Queue                                           |
|    - shutdown(wait=True, timeout=10.0)                           |
|    - Wait for pending audit events                               |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
| 6. Close Database Connections                                     |
|    - db.close_all()                                               |
|    - Dispose all per-loop engines                                |
|    - Close all Cloud SQL connectors                              |
+------------------------------------------------------------------+
```

---

## 10. Performance Impact Summary

### Latency Improvements

| Pattern | Before | After | Improvement |
|---------|--------|-------|-------------|
| Sequential cache checks | 300ms | 100ms | 3x faster |
| Blocking GCS I/O | Blocks loop | Non-blocking | 100+ concurrent |
| No file caching | 100-500ms/file | ~1ms (cached) | 100-500x faster |
| No response cache | Full LLM call | ~0ms (cached) | 1000x+ faster |
| Sequential DB lookups | N * 10ms | max(N * 10ms) | N times faster |
| Sync audit logging | +50ms/request | +0ms/request | Eliminated |
| Cold DB connections | +1.2s first call | ~0ms (pre-warmed) | Eliminated |

### Throughput Improvements

| Component | Single-Threaded | With Pools | Improvement |
|-----------|-----------------|------------|-------------|
| Agent invocations | 1 concurrent | 10 concurrent | 10x |
| GCS operations | 1 concurrent | 20 concurrent | 20x |
| SQL queries | 1 concurrent | 10 concurrent | 10x |
| Bulk document processing | 1 sequential | N concurrent | N times faster |
| Overall requests | ~10 RPS | 100+ RPS | 10x+ |

### Resource Efficiency

| Resource | Strategy | Impact |
|----------|----------|--------|
| Database connections | Pooled (5+10) | Reuse eliminates handshake |
| Memory | LRU caches with limits | Bounded memory usage |
| CPU | Async I/O | No wasted cycles on I/O wait |
| Threads | Workload-specific pools | No starvation |

---

## Key Files Reference

| Component | File |
|-----------|------|
| Executor Pools | `src/core/executors.py` |
| Async Utilities | `src/utils/async_utils.py` |
| Database Connection | `src/db/connection.py` |
| GCS Cache | `src/agents/document/gcs_cache.py` |
| File Cache | `src/agents/sheets/cache.py` |
| Session Manager | `src/agents/core/session_manager.py` |
| Rate Limiter | `src/agents/core/rate_limiter.py` |
| Background Queue Base | `src/core/queues/base_queue.py` |
| Audit Queue | `src/agents/core/audit_queue.py` |
| Usage Queue | `src/core/usage/usage_queue.py` |
| Bulk Job Queue | `src/bulk/queue.py` |
| GCS Storage | `src/storage/gcs.py` |
| App Lifecycle | `src/api/app.py` |
| Quota Cache | `src/core/usage/quota_checker.py` |
| Tier Cache | `src/core/usage/subscription_manager.py` |
| Store Cache | `src/rag/gemini_file_store.py` |

---

## Configuration Reference

### Environment Variables

```bash
# Executor Pool Sizes
AGENT_EXECUTOR_POOL_SIZE=10       # LLM agent invocations
IO_EXECUTOR_POOL_SIZE=20          # GCS/file I/O operations
QUERY_EXECUTOR_POOL_SIZE=10       # DuckDB/SQL queries

# Database Pool (per-loop differentiation)
DB_POOL_SIZE=3                    # Main loop pool size
DB_BACKGROUND_POOL_SIZE=2         # Background loop pool size
DB_MAX_OVERFLOW=5                 # Overflow connections
DB_POOL_TIMEOUT=30                # Connection acquire timeout (seconds)
DB_POOL_RECYCLE=1800              # Connection recycle time (30 minutes)

# Rate Limiting
RATE_LIMIT_REQUESTS=10            # Max requests per window
RATE_LIMIT_WINDOW=60              # Window size (seconds)

# Session
SESSION_TIMEOUT_MINUTES=30        # Session inactivity timeout

# Cleanup
CLEANUP_INTERVAL_SECONDS=300      # Periodic cleanup interval (5 minutes)

# Cache TTLs
QUOTA_CACHE_TTL_SECONDS=60        # Quota check cache TTL
TIER_CACHE_TTL_SECONDS=3600       # Subscription tier cache TTL (1 hour)
STORE_CACHE_TTL_SECONDS=300       # File store metadata cache TTL (5 minutes)

# Bulk Processing
BULK_CONCURRENT_DOCUMENTS=5       # Max concurrent documents in bulk queue
```
