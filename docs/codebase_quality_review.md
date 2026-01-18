# Codebase Quality Review: God Files, Duplicate Code, Dead Code & Hardcoded Values

**Review Date:** 2026-01-18
**Status:** Phase 1 Complete, Phases 2-5 Pending

---

## Executive Summary

A comprehensive review of the codebase identified significant technical debt across four categories:
- **16 god files** (>500 lines) requiring refactoring
- **10+ duplicate code patterns** affecting maintainability
- **40+ hardcoded values** that should be configurable
- **Several instances of dead/deprecated code**

---

## 1. God Files (Files >500 Lines with Mixed Responsibilities)

### Critical Priority (Immediate Attention)

| File | Lines | Issue |
|------|-------|-------|
| `src/agents/document/core.py` | **1,786** | Single class with 15+ concerns: LLM init, tool selection, caching, context prep, audit logging, token tracking |
| `src/db/repositories/bulk_repository.py` | 929 | 26 flat functions, no class encapsulation |
| `src/api/routers/audit.py` | 824 | 6 mixed concerns: dashboard, jobs, documents, audit, activity |

### High Priority

| File | Lines | Issue |
|------|-------|-------|
| `src/api/routers/rag.py` | 796 | Store + folder management mixed |
| `src/db/repositories/extraction_repository.py` | 777 | Schema generation + persistence mixed |
| `src/db/repositories/rag_repository.py` | 777 | No class encapsulation |
| `src/agents/extractor/core.py` | 777 | Monolithic agent, complex init |
| `src/api/routers/documents.py` | 756 | 8 endpoints with repeated patterns |
| `src/api/routers/bulk.py` | 696 | 4 concerns: folders, jobs, webhooks, upload |
| `src/agents/sheets/core.py` | 682 | Tool + cache + session mixed |
| `src/agents/sheets/tools.py` | 659 | 4 tools + DuckDB logic in one file |

### Medium Priority

| File | Lines | Issue |
|------|-------|-------|
| `src/api/routers/ingest.py` | 665 | Well-focused but large |
| `src/bulk/state_graph.py` | 602 | All node handlers in one file |
| `src/bulk/service.py` | 573 | Cache + orchestration mixed |
| `src/api/app.py` | 536 | Middleware + routing mixed |

---

## 2. Duplicate Code Patterns

### Critical Duplications

#### 2.1 Endpoint Error Handling (10+ occurrences)
**Files:** `documents.py`, `sheets.py`, `extraction/*.py`

Same pattern repeated:
```python
start_time = time.time()
try:
    # endpoint logic
    return Response(success=True, ...)
except Exception as e:
    logger.exception(f"... failed: {e}")
    return Response(success=False, error=str(e), processing_time_ms=elapsed_ms(start_time))
```

**Recommendation:** Create `@with_endpoint_timing` decorator

#### 2.2 GCS Cache Functions (3 identical implementations)
**File:** `src/agents/document/gcs_cache.py:179-344`
- `check_and_read_cached_summary()` (lines 179-223)
- `check_and_read_cached_faqs()` (lines 226-278)
- `check_and_read_cached_questions()` (lines 281-344)

**Recommendation:** Create generic `check_and_read_cached_content(content_type, parser_fn)`

#### 2.3 Generation Tool Cache Logic (3 tools)
**Files:** `summary_generator.py:56-80`, `faq_generator.py:56-80`, `question_generator.py`

**Recommendation:** Extract to base class `BaseGeneratorTool.check_cache()`

### Medium Duplications

#### 2.4 Database Session Pattern (94 occurrences)
```python
async with db.session() as session:
    stmt = select(Model).where(...)
    result = await session.execute(stmt)
```

**Files:** All repository files

**Recommendation:** Create base CRUD helpers: `get_by_id()`, `get_all()`, `upsert()`

#### 2.5 Token Usage Mapping (2 implementations)
**Files:**
- `src/api/utils/responses.py:76-102`
- `src/api/routers/extraction/helpers.py:52-69`

**Recommendation:** Use centralized `responses.py` version everywhere

#### 2.6 Document Endpoint Cache Pattern (3 times in documents.py)
Lines 151-168, 231-249, 312-338 - nearly identical cache check patterns

---

## 3. Hardcoded Values

### Critical (Must Fix) - **COMPLETED IN PHASE 1**

#### 3.1 Hardcoded Timeouts ✅ FIXED
| File | Line | Value | Fix Applied |
|------|------|-------|-------------|
| `document/core.py` | 229 | `timeout=300` | `Timeouts.LLM_EXECUTION` |
| `document/tools/faq_generator.py` | 43 | `timeout=300` | `Timeouts.LLM_EXECUTION` |
| `document/tools/summary_generator.py` | 43 | `timeout=300` | `Timeouts.LLM_EXECUTION` |
| `document/tools/question_generator.py` | 51 | `timeout=120` | `Timeouts.QUESTION_GENERATION` |

#### 3.2 Hardcoded Model Names with Versions (STILL PENDING)
| File | Line | Value |
|------|------|-------|
| `middleware/tool_selector.py` | 27 | `"gpt-5.2-2025-12-11"` |
| `middleware/stack.py` | 49, 278 | `"gpt-5-nano"` |
| `middleware/resilience.py` | 105 | `"gpt-5.2-2025-12-11"` |
| `rag/llama_parse_util.py` | 129, 132 | `"gemini-2.5-pro"`, `"openai-gpt-5-mini"` |

#### 3.3 Duplicate Constants ✅ FIXED
`STORE_CACHE_TTL_SECONDS = 300` - removed duplicate from `gemini_file_store.py`

### High Priority

#### 3.4 Magic Numbers ✅ FIXED
| File | Line | Value | Fix Applied |
|------|------|-------|-------------|
| `routers/content.py` | 109-110 | `3000` | `CHARS_PER_PAGE_ESTIMATE` |
| `routers/ingest.py` | 355-356 | `3000` | `CHARS_PER_PAGE_ESTIMATE` |

#### 3.5 Hardcoded Token Estimates in @check_quota ✅ FIXED
All endpoints now use `QuotaEstimates` class.

### Medium Priority (STILL PENDING)

#### 3.6 Inconsistent Chunking Defaults ✅ FIXED
Updated `gemini_file_store.py` to import from `constants.py`

---

## 4. Dead Code / Deprecated Code

### Unused Imports (STILL PENDING)
| File | Line | Import |
|------|------|--------|
| `document/core.py` | 23 | `from tenacity import retry, stop_after_attempt...` (not used, retry handled by middleware) |

### Deprecated Modules (Still in Use)
| File | Status |
|------|--------|
| `src/db/repositories/audit_repository.py` | DEPRECATED: Split into audit/ subpackage |
| `src/core/usage/models.py` | DEPRECATED: Re-exports from biz2bricks_core |

### Empty Stubs
| File | Lines | Issue |
|------|-------|-------|
| `core/usage/callback_handler.py` | 187, 191, 197, 201 | 4 consecutive `pass` stubs |
| `core/patterns/singleton.py` | 66, 94, 103 | Catch-and-pass exception handlers |

---

## 5. Recommended Refactoring Plan

### Phase 1: Quick Wins ✅ COMPLETED
1. ✅ **Consolidated hardcoded values into constants.py**
   - Added `CHARS_PER_PAGE_ESTIMATE = 3000`
   - Added `QuotaEstimates` class for token estimates

2. ✅ **Removed duplicate STORE_CACHE_TTL_SECONDS** from gemini_file_store.py

3. ✅ **Fixed chunking defaults** to use centralized constants

4. ✅ **Replaced hardcoded timeouts** with `Timeouts` class

5. ⏳ **Convert model names to env vars** - PENDING

6. ⏳ **Remove unused tenacity imports** - PENDING

### Phase 2: Extract Common Patterns (3-5 days) - PENDING
1. **Create endpoint error handling decorator**
   - File: `src/api/utils/decorators.py`
   - Replace 10+ duplicate patterns

2. **Create generic GCS cache function**
   - File: `src/agents/document/gcs_cache.py`
   - Replace 3 near-identical functions

3. **Consolidate token usage mapping**
   - Remove duplicate from `extraction/helpers.py`

### Phase 3: Repository Refactoring (3-5 days) - PENDING
1. **Convert bulk_repository.py to class-based**
   - Create `BulkRepository` class
   - Group operations by entity (JobOps, DocumentOps)

2. **Convert extraction_repository.py to class-based**
   - Extract SchemaGenerator for DDL logic

3. **Convert rag_repository.py to class-based**
   - Or split into StoreRepository + FolderRepository

### Phase 4: Router Splitting (2-3 days) - PENDING
1. **Split audit.py** into:
   - `dashboard.py`, `jobs.py`, `documents.py`, `activity.py`

2. **Split rag.py** into:
   - `stores.py`, `folders.py`

3. **Split bulk.py** into:
   - `folders.py`, `jobs.py`, `webhooks.py`

### Phase 5: Agent Refactoring (5-7 days) - PENDING
1. **DocumentAgent (1,786 lines)** - Extract:
   - `ToolManager` for tool initialization/selection
   - `ContextBuilder` for request context prep
   - `CacheHandler` for response caching
   - `AuditLogger` wrapper/decorator

2. **SheetsAgent** - Extract:
   - `ToolFactory` for tool initialization
   - `FileCacheManager` for cache operations

3. **Split sheets/tools.py** into:
   - `tools/file_preview.py`
   - `tools/cross_query.py`
   - `tools/smart_analysis.py`
   - `utils/duckdb_pool.py`

---

## 6. Verification Plan

After each phase:
1. Run full test suite: `pytest tests/`
2. Run with coverage: `pytest tests/ --cov=src`
3. Start server and test key endpoints manually
4. Check for import errors with: `python -c "from src.main import app"`

---

## 7. Files Modified in Phase 1

| File | Changes |
|------|---------|
| `src/constants.py` | Added `CHARS_PER_PAGE_ESTIMATE`, `QuotaEstimates` class |
| `src/rag/gemini_file_store.py` | Removed duplicate constant, import from constants, fix chunking |
| `src/api/routers/content.py` | Use `CHARS_PER_PAGE_ESTIMATE` |
| `src/api/routers/ingest.py` | Use `CHARS_PER_PAGE_ESTIMATE` |
| `src/agents/document/core.py` | Use `Timeouts.LLM_EXECUTION` |
| `src/agents/document/tools/summary_generator.py` | Use `Timeouts.LLM_EXECUTION` |
| `src/agents/document/tools/faq_generator.py` | Use `Timeouts.LLM_EXECUTION` |
| `src/agents/document/tools/question_generator.py` | Use `Timeouts.QUESTION_GENERATION` |
| `src/api/routers/documents.py` | Use `QuotaEstimates` (5 places) |
| `src/api/routers/sheets.py` | Use `QuotaEstimates.SHEETS_ANALYZE` |
| `src/api/routers/extraction/analyze.py` | Use `QuotaEstimates.EXTRACTION_ANALYZE` |
| `src/api/routers/extraction/schema.py` | Use `QuotaEstimates.EXTRACTION_SCHEMA` |
| `src/api/routers/extraction/extract.py` | Use `QuotaEstimates.EXTRACTION_EXTRACT` |

---

## 8. Priority Files for Next Phases

### Immediate (Phase 2)
- `src/api/utils/decorators.py` - Add endpoint error handling decorator
- `src/agents/document/gcs_cache.py` - Consolidate cache functions
- `src/api/routers/extraction/helpers.py` - Remove duplicate token mapping

### Short-term (Phase 3)
- `src/db/repositories/bulk_repository.py` - Convert to class
- `src/db/repositories/extraction_repository.py` - Convert to class

### Medium-term (Phases 4-5)
- `src/api/routers/audit.py` - Split into submodules
- `src/agents/document/core.py` - Extract ToolManager, ContextBuilder, etc.
