#!/usr/bin/env python3
"""
Fix stale relevance scores in the semantic cache.

Cached RAG query results from before the relevance score fix have
relevance_score: 0.0 for all citations. This script updates them
to 0.8 (default high confidence when citations exist).

Usage:
    python scripts/fix_cache_relevance.py [--org-id ORG_ID] [--clear]

Options:
    --org-id    Fix only for a specific organization (default: all)
    --clear     Delete stale entries instead of updating them
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from src.db.connection import db


async def fix_stale_relevance_scores(org_id: str | None = None) -> int:
    """Update cached citations with relevance_score=0 to 0.8."""
    async with db.session() as session:
        if session is None:
            print("ERROR: Database session unavailable")
            return 0

        try:
            # Build query based on whether org_id is specified
            if org_id:
                query = text("""
                    UPDATE rag_query_cache
                    SET citations = (
                        SELECT jsonb_agg(
                            CASE
                                WHEN (c->>'relevance_score')::float = 0
                                THEN jsonb_set(c, '{relevance_score}', '0.8')
                                ELSE c
                            END
                        )
                        FROM jsonb_array_elements(citations) AS c
                    )
                    WHERE org_id = :org_id
                      AND citations IS NOT NULL
                      AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(citations) AS c
                        WHERE (c->>'relevance_score')::float = 0
                      )
                """)
                result = await session.execute(query, {"org_id": org_id})
            else:
                query = text("""
                    UPDATE rag_query_cache
                    SET citations = (
                        SELECT jsonb_agg(
                            CASE
                                WHEN (c->>'relevance_score')::float = 0
                                THEN jsonb_set(c, '{relevance_score}', '0.8')
                                ELSE c
                            END
                        )
                        FROM jsonb_array_elements(citations) AS c
                    )
                    WHERE citations IS NOT NULL
                      AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(citations) AS c
                        WHERE (c->>'relevance_score')::float = 0
                      )
                """)
                result = await session.execute(query)

            updated = result.rowcount
            print(f"Updated {updated} cache entries with fixed relevance scores")
            return updated

        except Exception as e:
            print(f"ERROR: {e}")
            return 0


async def clear_stale_entries(org_id: str | None = None) -> int:
    """Delete cached entries with relevance_score=0."""
    async with db.session() as session:
        if session is None:
            print("ERROR: Database session unavailable")
            return 0

        try:
            if org_id:
                query = text("""
                    DELETE FROM rag_query_cache
                    WHERE org_id = :org_id
                      AND citations IS NOT NULL
                      AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(citations) AS c
                        WHERE (c->>'relevance_score')::float = 0
                      )
                """)
                result = await session.execute(query, {"org_id": org_id})
            else:
                query = text("""
                    DELETE FROM rag_query_cache
                    WHERE citations IS NOT NULL
                      AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(citations) AS c
                        WHERE (c->>'relevance_score')::float = 0
                      )
                """)
                result = await session.execute(query)

            deleted = result.rowcount
            print(f"Deleted {deleted} stale cache entries")
            return deleted

        except Exception as e:
            print(f"ERROR: {e}")
            return 0


async def main():
    parser = argparse.ArgumentParser(description="Fix stale relevance scores in semantic cache")
    parser.add_argument("--org-id", help="Organization ID to fix (default: all)")
    parser.add_argument("--clear", action="store_true", help="Delete stale entries instead of updating")
    args = parser.parse_args()

    print("Connecting to database...")

    if args.clear:
        print(f"Clearing stale cache entries{' for org ' + args.org_id if args.org_id else ''}...")
        count = await clear_stale_entries(args.org_id)
    else:
        print(f"Fixing relevance scores{' for org ' + args.org_id if args.org_id else ''}...")
        count = await fix_stale_relevance_scores(args.org_id)

    print(f"Done! Affected {count} entries.")


if __name__ == "__main__":
    asyncio.run(main())
