"""
Semantic cache repository - PostgreSQL implementation for RAG query caching.

Uses pgvector for semantic similarity search to find cached responses
for semantically similar queries, reducing LLM and vector search costs.

Multi-tenancy: All operations are scoped by organization_id.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert

from ..models import RAGQueryCache, PGVECTOR_AVAILABLE
from ..connection import db
from ..utils import with_db_retry
from src.utils.env_utils import parse_bool_env, parse_int_env, parse_float_env

logger = logging.getLogger(__name__)

# Configuration
SIMILARITY_THRESHOLD = parse_float_env("SEMANTIC_CACHE_THRESHOLD", 0.85)
CACHE_TTL_HOURS = parse_int_env("SEMANTIC_CACHE_TTL_HOURS", 24)
CACHE_ENABLED = parse_bool_env("SEMANTIC_CACHE_ENABLED", True)


def is_cache_enabled() -> bool:
    """Check if semantic cache is enabled and available."""
    return CACHE_ENABLED and PGVECTOR_AVAILABLE and db.config.enabled


# =============================================================================
# CACHE LOOKUP OPERATIONS
# =============================================================================


@with_db_retry
async def find_similar_query(
    org_id: str,
    query_embedding: List[float],
    folder_filter: Optional[str] = None,
    file_filter: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Find cached answer for a semantically similar query using vector similarity.

    Uses cosine similarity via pgvector's <=> operator.
    Only returns results above the similarity threshold (default 85%).

    Args:
        org_id: Organization ID for tenant isolation
        query_embedding: 768-dimensional embedding vector from Gemini
        folder_filter: Optional folder name to scope cache lookup
        file_filter: Optional file name to scope cache lookup
        threshold: Similarity threshold (0-1), defaults to SIMILARITY_THRESHOLD

    Returns:
        Dict with cached answer, citations, and similarity score, or None if no match
    """
    if not is_cache_enabled():
        return None

    threshold = threshold or SIMILARITY_THRESHOLD

    async with db.session() as session:
        if session is None:
            return None

        try:
            # Convert embedding list to PostgreSQL vector format
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            # Build the query with optional filters
            # Using 1 - (embedding <=> query) for cosine similarity
            # SQLAlchemy text() requires :param style with dict params
            query = text("""
                SELECT
                    id,
                    query_text,
                    answer,
                    citations,
                    folder_filter,
                    file_filter,
                    search_mode,
                    hit_count,
                    created_at,
                    1 - (query_embedding <=> CAST(:embedding AS vector)) as similarity
                FROM rag_query_cache
                WHERE org_id = :org_id
                  AND folder_filter IS NOT DISTINCT FROM :folder_filter
                  AND file_filter IS NOT DISTINCT FROM :file_filter
                  AND 1 - (query_embedding <=> CAST(:embedding AS vector)) > :threshold
                ORDER BY similarity DESC
                LIMIT 1
            """)

            result = await session.execute(
                query,
                {
                    "embedding": embedding_str,
                    "org_id": org_id,
                    "folder_filter": folder_filter,
                    "file_filter": file_filter,
                    "threshold": threshold,
                }
            )

            row = result.fetchone()

            if row:
                # Increment hit count
                await session.execute(
                    text("""
                        UPDATE rag_query_cache
                        SET hit_count = hit_count + 1
                        WHERE id = CAST(:cache_id AS uuid)
                    """),
                    {"cache_id": str(row.id)}
                )

                logger.info(
                    f"Semantic cache HIT: similarity={row.similarity:.3f}, "
                    f"original_query='{row.query_text[:50]}...'"
                )

                return {
                    "id": str(row.id),
                    "query_text": row.query_text,
                    "answer": row.answer,
                    "citations": row.citations or [],
                    "folder_filter": row.folder_filter,
                    "file_filter": row.file_filter,
                    "search_mode": row.search_mode,
                    "hit_count": row.hit_count + 1,
                    "created_at": row.created_at,
                    "similarity": float(row.similarity),
                }

            logger.debug(f"Semantic cache MISS for org {org_id}")
            return None

        except Exception as e:
            logger.error(f"Error in semantic cache lookup: {e}")
            return None


# =============================================================================
# CACHE STORAGE OPERATIONS
# =============================================================================


@with_db_retry
async def cache_query(
    org_id: str,
    query_text: str,
    query_embedding: List[float],
    answer: str,
    citations: Optional[List[Dict[str, Any]]] = None,
    folder_filter: Optional[str] = None,
    file_filter: Optional[str] = None,
    search_mode: str = "hybrid",
) -> Optional[str]:
    """
    Store a query and its answer in the semantic cache.

    Args:
        org_id: Organization ID for tenant isolation
        query_text: Original query text
        query_embedding: 768-dimensional embedding vector
        answer: Generated answer to cache
        citations: List of citation dicts
        folder_filter: Folder filter used in the query
        file_filter: File filter used in the query
        search_mode: Search mode used (semantic/keyword/hybrid)

    Returns:
        Cache entry ID if stored, None if caching disabled or failed
    """
    if not is_cache_enabled():
        return None

    async with db.session() as session:
        if session is None:
            return None

        try:
            cache_id = str(uuid4())
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
            citations_json = json.dumps(citations) if citations else None

            # Insert into cache using named parameters
            await session.execute(
                text("""
                    INSERT INTO rag_query_cache (
                        id, org_id, query_text, query_embedding,
                        answer, citations, folder_filter, file_filter,
                        search_mode, created_at, hit_count
                    ) VALUES (
                        CAST(:cache_id AS uuid), :org_id, :query_text, CAST(:embedding AS vector),
                        :answer, CAST(:citations AS jsonb), :folder_filter, :file_filter,
                        :search_mode, :created_at, 0
                    )
                """),
                {
                    "cache_id": cache_id,
                    "org_id": org_id,
                    "query_text": query_text,
                    "embedding": embedding_str,
                    "answer": answer,
                    "citations": citations_json,
                    "folder_filter": folder_filter,
                    "file_filter": file_filter,
                    "search_mode": search_mode,
                    "created_at": datetime.utcnow(),
                }
            )

            logger.info(f"Cached query for org {org_id}: '{query_text[:50]}...'")
            return cache_id

        except Exception as e:
            logger.error(f"Error caching query: {e}")
            return None


# =============================================================================
# CACHE MANAGEMENT OPERATIONS
# =============================================================================


@with_db_retry
async def get_cache_stats(org_id: str) -> Dict[str, Any]:
    """
    Get cache statistics for an organization.

    Returns:
        Dict with cache entry count, total hits, and oldest entry age
    """
    if not is_cache_enabled():
        return {"enabled": False, "entries": 0, "total_hits": 0}

    async with db.session() as session:
        if session is None:
            return {"enabled": False, "entries": 0, "total_hits": 0}

        try:
            result = await session.execute(
                text("""
                    SELECT
                        COUNT(*) as entry_count,
                        COALESCE(SUM(hit_count), 0) as total_hits,
                        MIN(created_at) as oldest_entry
                    FROM rag_query_cache
                    WHERE org_id = :org_id
                """),
                {"org_id": org_id}
            )

            row = result.fetchone()

            return {
                "enabled": True,
                "entries": row.entry_count,
                "total_hits": int(row.total_hits),
                "oldest_entry": row.oldest_entry,
                "similarity_threshold": SIMILARITY_THRESHOLD,
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"enabled": True, "error": str(e)}


@with_db_retry
async def clear_cache(
    org_id: str,
    folder_filter: Optional[str] = None,
    file_filter: Optional[str] = None,
) -> int:
    """
    Clear cache entries for an organization, optionally filtered by folder/file.

    Args:
        org_id: Organization ID
        folder_filter: Optional - only clear entries for this folder
        file_filter: Optional - only clear entries for this file

    Returns:
        Number of entries deleted
    """
    if not is_cache_enabled():
        return 0

    async with db.session() as session:
        if session is None:
            return 0

        try:
            if folder_filter:
                result = await session.execute(
                    text("""
                        DELETE FROM rag_query_cache
                        WHERE org_id = :org_id AND folder_filter = :folder_filter
                    """),
                    {"org_id": org_id, "folder_filter": folder_filter}
                )
            elif file_filter:
                result = await session.execute(
                    text("""
                        DELETE FROM rag_query_cache
                        WHERE org_id = :org_id AND file_filter = :file_filter
                    """),
                    {"org_id": org_id, "file_filter": file_filter}
                )
            else:
                result = await session.execute(
                    text("""
                        DELETE FROM rag_query_cache
                        WHERE org_id = :org_id
                    """),
                    {"org_id": org_id}
                )

            deleted = result.rowcount
            logger.info(f"Cleared {deleted} cache entries for org {org_id}")
            return deleted

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0


@with_db_retry
async def cleanup_expired_entries(ttl_hours: Optional[int] = None) -> int:
    """
    Remove cache entries older than TTL.

    Args:
        ttl_hours: Hours after which entries expire (default: CACHE_TTL_HOURS)

    Returns:
        Number of entries deleted
    """
    if not is_cache_enabled():
        return 0

    ttl = ttl_hours or CACHE_TTL_HOURS

    async with db.session() as session:
        if session is None:
            return 0

        try:
            result = await session.execute(
                text(f"""
                    DELETE FROM rag_query_cache
                    WHERE created_at < NOW() - INTERVAL '{ttl} hours'
                """)
            )

            deleted = result.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} expired cache entries (TTL: {ttl}h)")
            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up expired entries: {e}")
            return 0
