#!/usr/bin/env python3
"""
Clean all data script - Fresh start for Document Intelligence AI.

Clears all data EXCEPT core user/org structure:
- PRESERVES: organizations, users, folders, file_search_stores, subscription_tiers, organization_subscriptions
- CLEARS: documents, audit_logs, processing_jobs, document_generations, bulk_jobs, etc.
- CLEARS: GCS bucket files
- CLEARS: Gemini File Search store documents

Usage:
    python scripts/clean_all_data.py              # Interactive mode (with confirmations)
    python scripts/clean_all_data.py --force      # Skip confirmations
    python scripts/clean_all_data.py --dry-run    # Show what would be deleted
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Tables to TRUNCATE (in order due to foreign key constraints)
TABLES_TO_CLEAR = [
    "bulk_job_documents",
    "bulk_jobs",
    "document_generations",
    "processing_jobs",
    "audit_logs",
    "documents",
    "document_folders",
    "rag_query_cache",
    "token_usage_records",
    "resource_usage_records",
    "usage_aggregations",
    "conversation_summaries",
    "memory_entries",
    "user_preferences",
    "sessions",
]

# Tables to PRESERVE
TABLES_TO_PRESERVE = [
    "organizations",
    "users",
    "folders",
    "file_search_stores",
    "subscription_tiers",
    "organization_subscriptions",
]


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

async def get_connection(database: str = "postgres"):
    """Get async database connection using Cloud SQL Connector or direct connection."""
    instance_name = os.getenv(
        "CLOUD_SQL_INSTANCE",
        "biz2bricks-dev-v1:us-central1:biz-2-bricks-intelli-doc-dev"
    )
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_password = os.getenv("DATABASE_PASSWORD", "")
    use_connector = os.getenv("USE_CLOUD_SQL_CONNECTOR", "true").lower() == "true"
    ip_type_str = os.getenv("CLOUD_SQL_IP_TYPE", "PUBLIC").upper()

    conn = None
    connector = None

    if use_connector:
        try:
            from google.cloud.sql.connector import Connector, IPTypes

            ip_type = IPTypes.PUBLIC if ip_type_str == "PUBLIC" else IPTypes.PRIVATE

            loop = asyncio.get_running_loop()
            connector = Connector(loop=loop)

            conn = await connector.connect_async(
                instance_name,
                "asyncpg",
                user=db_user,
                password=db_password,
                db=database,
                ip_type=ip_type,
            )
            return conn, connector
        except Exception as e:
            logger.warning(f"Cloud SQL Connector failed: {e}, falling back to direct connection")
            if connector:
                connector.close()

    # Direct connection fallback
    import asyncpg

    db_url = os.getenv("DATABASE_URL", "")
    if db_url and "@" in db_url:
        parts = db_url.split("@")
        host_part = parts[1].split("/")[0]
        host, port = host_part.split(":") if ":" in host_part else (host_part, "5432")
    else:
        host, port = "localhost", "5432"

    conn = await asyncpg.connect(
        host=host,
        port=int(port),
        user=db_user,
        password=db_password,
        database=database,
    )
    return conn, None


async def close_connection(conn, connector):
    """Close database connection and connector."""
    if conn:
        try:
            await conn.close()
        except Exception:
            pass
    if connector:
        try:
            connector.close()
        except Exception:
            pass


# =============================================================================
# GEMINI FILE SEARCH STORE CLEANUP
# =============================================================================

def clear_gemini_stores(dry_run: bool = False) -> int:
    """
    Clear all documents from all Gemini File Search stores.

    Args:
        dry_run: If True, only log what would be deleted

    Returns:
        Number of documents deleted
    """
    logger.info("=" * 60)
    logger.info("Step 1: Clearing Gemini File Search Store Contents")
    logger.info("=" * 60)

    try:
        from google import genai

        client = genai.Client()
        total_deleted = 0

        # List all stores
        stores = list(client.file_search_stores.list())
        logger.info(f"Found {len(stores)} Gemini File Search stores")

        for store in stores:
            logger.info(f"\nStore: {store.display_name} ({store.name})")

            # List all documents in this store
            documents = list(client.file_search_stores.documents.list(parent=store.name))
            logger.info(f"  Found {len(documents)} documents")

            if dry_run:
                for doc in documents:
                    logger.info(f"  [DRY-RUN] Would delete: {doc.display_name}")
                total_deleted += len(documents)
            else:
                for doc in documents:
                    try:
                        client.file_search_stores.documents.delete(
                            name=doc.name,
                            config={"force": True}
                        )
                        logger.info(f"  Deleted: {doc.display_name}")
                        total_deleted += 1
                    except Exception as e:
                        logger.error(f"  Failed to delete {doc.display_name}: {e}")

        logger.info(f"\nTotal documents {'would be ' if dry_run else ''}deleted from Gemini: {total_deleted}")
        return total_deleted

    except ImportError:
        logger.warning("Gemini genai module not available, skipping store cleanup")
        return 0
    except Exception as e:
        logger.error(f"Gemini store cleanup failed: {e}")
        return 0


# =============================================================================
# GCS BUCKET CLEANUP
# =============================================================================

def clear_gcs_bucket(dry_run: bool = False) -> int:
    """
    Clear all files from the GCS bucket.

    Args:
        dry_run: If True, only log what would be deleted

    Returns:
        Number of files deleted
    """
    logger.info("=" * 60)
    logger.info("Step 2: Clearing GCS Bucket Files")
    logger.info("=" * 60)

    bucket_name = os.getenv("GCS_BUCKET", "biz2bricks-dev-v1-document-store")
    logger.info(f"Target bucket: gs://{bucket_name}")

    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List all blobs
        blobs = list(bucket.list_blobs())
        logger.info(f"Found {len(blobs)} files in bucket")

        if dry_run:
            # Just count and show samples
            for i, blob in enumerate(blobs[:10]):
                logger.info(f"  [DRY-RUN] Would delete: {blob.name}")
            if len(blobs) > 10:
                logger.info(f"  ... and {len(blobs) - 10} more files")
            return len(blobs)

        # Delete all blobs
        deleted = 0
        for blob in blobs:
            try:
                blob.delete()
                deleted += 1
                if deleted % 100 == 0:
                    logger.info(f"  Deleted {deleted}/{len(blobs)} files...")
            except Exception as e:
                logger.error(f"  Failed to delete {blob.name}: {e}")

        logger.info(f"\nTotal files deleted from GCS: {deleted}")
        return deleted

    except ImportError:
        logger.warning("google-cloud-storage not available, skipping GCS cleanup")
        return 0
    except Exception as e:
        logger.error(f"GCS bucket cleanup failed: {e}")
        return 0


# =============================================================================
# DATABASE CLEANUP
# =============================================================================

async def clear_database_tables(dry_run: bool = False) -> dict:
    """
    Clear all data from database tables (preserving core tables).

    Args:
        dry_run: If True, only show table row counts

    Returns:
        Dict with table names and rows deleted
    """
    logger.info("=" * 60)
    logger.info("Step 3: Clearing Database Tables")
    logger.info("=" * 60)

    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    logger.info(f"Target database: {target_db}")

    conn, connector = await get_connection(target_db)
    results = {}

    try:
        # First, show current row counts for all tables
        logger.info("\nCurrent table row counts:")
        logger.info("-" * 40)

        for table in TABLES_TO_CLEAR:
            try:
                count = await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')
                results[table] = {"before": count, "cleared": False}
                logger.info(f"  {table}: {count} rows")
            except Exception as e:
                logger.warning(f"  {table}: not found or error ({e})")
                results[table] = {"before": 0, "error": str(e)}

        logger.info("\nPreserved tables (will NOT be cleared):")
        for table in TABLES_TO_PRESERVE:
            try:
                count = await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')
                logger.info(f"  {table}: {count} rows [PRESERVED]")
            except Exception as e:
                logger.info(f"  {table}: not found ({e})")

        if dry_run:
            logger.info("\n[DRY-RUN] Would truncate the above tables")
            return results

        # Truncate tables in order
        logger.info("\nTruncating tables...")
        for table in TABLES_TO_CLEAR:
            try:
                await conn.execute(f'TRUNCATE TABLE "{table}" CASCADE')
                results[table]["cleared"] = True
                logger.info(f"  Truncated: {table}")
            except Exception as e:
                logger.warning(f"  Failed to truncate {table}: {e}")
                results[table]["error"] = str(e)

        # Reset usage counters in preserved tables
        logger.info("\nResetting usage counters...")

        try:
            await conn.execute("""
                UPDATE organization_subscriptions SET
                    tokens_used_this_period = 0,
                    llamaparse_pages_used = 0,
                    file_search_queries_used = 0,
                    storage_used_bytes = 0
            """)
            logger.info("  Reset organization_subscriptions counters")
        except Exception as e:
            logger.warning(f"  Failed to reset organization_subscriptions: {e}")

        try:
            await conn.execute("""
                UPDATE file_search_stores SET
                    active_documents_count = 0,
                    total_size_bytes = 0
            """)
            logger.info("  Reset file_search_stores counters")
        except Exception as e:
            logger.warning(f"  Failed to reset file_search_stores: {e}")

        logger.info("\nDatabase cleanup complete!")
        return results

    finally:
        await close_connection(conn, connector)


# =============================================================================
# MAIN
# =============================================================================

async def run_cleanup(force: bool = False, dry_run: bool = False):
    """Run the complete cleanup process."""

    logger.info("\n" + "=" * 60)
    logger.info("Document Intelligence AI - Data Cleanup")
    logger.info("=" * 60)

    if dry_run:
        logger.info("MODE: DRY-RUN (no actual deletions)")
    else:
        logger.info("MODE: LIVE (data WILL be deleted)")

    logger.info("\nWill CLEAR:")
    logger.info("  - Gemini File Search store documents")
    logger.info("  - GCS bucket files")
    logger.info(f"  - Database tables: {', '.join(TABLES_TO_CLEAR)}")

    logger.info("\nWill PRESERVE:")
    logger.info(f"  - Database tables: {', '.join(TABLES_TO_PRESERVE)}")

    if not force and not dry_run:
        print("\n" + "!" * 60)
        print("WARNING: This will DELETE all document data, audit logs,")
        print("         usage history, and files. This cannot be undone!")
        print("!" * 60)
        confirm = input("\nAre you sure you want to proceed? (type 'yes' to confirm): ")
        if confirm.lower() != "yes":
            logger.info("Operation cancelled by user")
            return

    # Step 1: Clear Gemini stores
    gemini_deleted = clear_gemini_stores(dry_run=dry_run)

    # Step 2: Clear GCS bucket
    gcs_deleted = clear_gcs_bucket(dry_run=dry_run)

    # Step 3: Clear database tables
    db_results = await clear_database_tables(dry_run=dry_run)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Cleanup Summary")
    logger.info("=" * 60)
    logger.info(f"Gemini documents {'would be ' if dry_run else ''}deleted: {gemini_deleted}")
    logger.info(f"GCS files {'would be ' if dry_run else ''}deleted: {gcs_deleted}")

    total_db_rows = sum(r.get("before", 0) for r in db_results.values())
    logger.info(f"Database rows {'would be ' if dry_run else ''}cleared: {total_db_rows}")

    if not dry_run:
        logger.info("\n" + "-" * 60)
        logger.info("IMPORTANT: Users will need to re-login (sessions cleared)")
        logger.info("-" * 60)

    logger.info("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description="Clean all data for a fresh start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/clean_all_data.py              # Interactive mode
  python scripts/clean_all_data.py --dry-run    # Preview what would be deleted
  python scripts/clean_all_data.py --force      # Skip confirmation
        """
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    asyncio.run(run_cleanup(force=args.force, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
