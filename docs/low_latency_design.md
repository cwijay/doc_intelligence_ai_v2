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
8. [Lifecycle Management](#8-lifecycle-management)
9. [Performance Impact Summary](#9-performance-impact-summary)

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
|  +------------------+                    +------------------+                     |
|  |   Audit Queue    |                    |   Usage Queue    |                     |
|  | (dedicated loop) |                    | (dedicated loop) |                     |
|  +--------+---------+                    +--------+---------+                     |
|           |                                       |                               |
|           v                                       v                               |
|  +--------+---------+                    +--------+---------+                     |
|  |   PostgreSQL     |<------------------>|   PostgreSQL     |                     |
|  | (async pooled)   |                    | (async pooled)   |                     |
|  +------------------+                    +------------------+                     |
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
| File/DataFrame | LRU | Unlimited | 50 files | `core.py` |
| GCS Content | Persistent | Unlimited | GCS storage | `gcs_cache.py` |
| Tier Config | TTL | 1 hour | 10 tiers | `subscription_manager.py` |
| Store Metadata | TTL | 5 minutes | Unlimited | `gemini_file_store.py` |

---

## 5. Connection Pooling

### SQLAlchemy Async Pool

**File:** `src/db/connection.py`

```
+------------------------------------------------------------------+
|                    DatabaseManager                                |
|                      (Singleton)                                  |
+------------------------------------------------------------------+
|                                                                  |
|  Per-Event-Loop Engine Management:                               |
|  _engines: Dict[loop_id, AsyncEngine]                            |
|  _session_factories: Dict[loop_id, async_sessionmaker]           |
|                                                                  |
|  +------------------------+                                       |
|  |   AsyncAdaptedQueuePool |                                      |
|  |   - pool_size: 5        |                                      |
|  |   - max_overflow: 10    |                                      |
|  |   - pool_timeout: 30s   |                                      |
|  |   - pool_recycle: 30min |                                      |
|  +------------------------+                                       |
|              |                                                    |
|              v                                                    |
|  +----------+----------+----------+----------+----------+         |
|  | Conn 1   | Conn 2   | Conn 3   | Conn 4   | Conn 5   |         |
|  | (idle)   | (in use) | (idle)   | (in use) | (idle)   |         |
|  +----------+----------+----------+----------+----------+         |
|                         +                                         |
|              +----------+----------+                              |
|              | Overflow connections (up to 10)                   |
|              +----------+----------+                              |
+------------------------------------------------------------------+
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

### Architecture

```
+------------------------------------------------------------------+
|                     MAIN EVENT LOOP                               |
|                    (API Requests)                                 |
+------------------------------------------------------------------+
       |                                      |
       | enqueue_audit_event()                | enqueue_usage_event()
       | (non-blocking)                       | (non-blocking)
       v                                      v
+------------------+                  +------------------+
| Queue.Queue      |                  | Queue.Queue      |
| (max_size: 1000) |                  | (max_size: 1000) |
+--------+---------+                  +--------+---------+
         |                                     |
         | (thread boundary)                   | (thread boundary)
         v                                     v
+------------------+                  +------------------+
| Audit Thread     |                  | Usage Thread     |
| - Own event loop |                  | - Own event loop |
| - Own DB session |                  | - Own DB session |
+--------+---------+                  +--------+---------+
         |                                     |
         v                                     v
+------------------+                  +------------------+
|   PostgreSQL     |                  |   PostgreSQL     |
| (async session)  |                  | (async session)  |
+------------------+                  +------------------+
```

### Event Flow

```
API Request Thread                   Background Worker Thread
       |                                      |
       | AuditEvent(                          |
       |   type="query",                      |
       |   details={...}                      |
       | )                                    |
       |                                      |
       | queue.put(event)  ------------------>| queue.get(timeout=1.0)
       |                                      |
       | return response   (non-blocking)     | await save_to_db(event)
       |                                      |
       v                                      v
  [Response sent]                      [Event persisted]
  (0ms overhead)                       (async in background)
```

### Graceful Shutdown

```
Shutdown Signal
       |
       v
+------------------+
| Send poison pill |---> queue.put(None)
+------------------+
       |
       v
+------------------+
| Wait for drain   |---> thread.join(timeout=30)
+------------------+
       |
       v
+------------------+
| Close DB session |---> await session.close()
+------------------+
       |
       v
+------------------+
| Stop event loop  |---> loop.stop()
+------------------+
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

## 8. Lifecycle Management

### Startup Sequence

```
+------------------------------------------------------------------+
|                     APPLICATION STARTUP                           |
+------------------------------------------------------------------+
                              |
                              v
+------------------+  +------------------+  +------------------+
| 1. Init DB Pool  |->| 2. Start Queues  |->| 3. Init Agents   |
| - Create engine  |  | - Audit thread   |  | - LLM clients    |
| - Session factory|  | - Usage thread   |  | - Tools          |
| - Test connection|  | - Start loops    |  | - Middleware     |
+------------------+  +------------------+  +------------------+
                              |
                              v
               +------------------+
               | 4. Start Cleanup |
               | - asyncio.task   |
               | - 5 min interval |
               +------------------+
                              |
                              v
               +------------------+
               | 5. Ready to Serve|
               +------------------+
```

### Shutdown Sequence

```
+------------------------------------------------------------------+
|                    APPLICATION SHUTDOWN                           |
+------------------------------------------------------------------+
                              |
                              v
+------------------+  +------------------+  +------------------+
| 1. Cancel Tasks  |->| 2. Shutdown     |->| 3. Drain Queues  |
| - Cleanup task   |  |    Agents       |  | - Audit queue    |
|                  |  | - Clear caches  |  | - Usage queue    |
|                  |  | - Close DuckDB  |  | - Wait 30s max   |
+------------------+  +------------------+  +------------------+
                              |
                              v
+------------------+  +------------------+
| 4. Shutdown      |->| 5. Close DB     |
|    Executors     |  | - All pools     |
| - Wait for tasks |  | - All engines   |
+------------------+  +------------------+
```

---

## 9. Performance Impact Summary

### Latency Improvements

| Pattern | Before | After | Improvement |
|---------|--------|-------|-------------|
| Sequential cache checks | 300ms | 100ms | 3x faster |
| Blocking GCS I/O | Blocks loop | Non-blocking | 100+ concurrent |
| No file caching | 100-500ms/file | ~1ms (cached) | 100-500x faster |
| No response cache | Full LLM call | ~0ms (cached) | 1000x+ faster |
| Sequential DB lookups | N * 10ms | max(N * 10ms) | N times faster |
| Sync audit logging | +50ms/request | +0ms/request | Eliminated |

### Throughput Improvements

| Component | Single-Threaded | With Pools | Improvement |
|-----------|-----------------|------------|-------------|
| Agent invocations | 1 concurrent | 10 concurrent | 10x |
| GCS operations | 1 concurrent | 20 concurrent | 20x |
| SQL queries | 1 concurrent | 10 concurrent | 10x |
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
| Session Manager | `src/agents/core/session_manager.py` |
| Rate Limiter | `src/agents/core/rate_limiter.py` |
| Background Queues | `src/core/queues/base_queue.py` |
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
AGENT_EXECUTOR_POOL_SIZE=10
IO_EXECUTOR_POOL_SIZE=20
QUERY_EXECUTOR_POOL_SIZE=10

# Database Pool
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=1800

# Rate Limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW_SECONDS=60

# Session
SESSION_TIMEOUT_MINUTES=30

# Cleanup
CLEANUP_INTERVAL_SECONDS=300

# Cache TTLs
QUOTA_CACHE_TTL_SECONDS=60
TIER_CACHE_TTL_SECONDS=3600
STORE_CACHE_TTL_SECONDS=300
```
