#!/usr/bin/env python3
"""
Migration script to add document status tracking columns.

This script adds the following columns to the documents table:
- organization_id: Multi-tenancy support
- folder_name: Document organization
- status: Track upload/parse lifecycle (uploaded, parsed, failed)
- parsed_path: Path to parsed .md file
- parsed_at: Timestamp when parsing completed

Usage:
    python scripts/migrate_documents_status.py          # Run migration
    python scripts/migrate_documents_status.py --dry-run # Show SQL without executing
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


# =============================================================================
# MIGRATION SQL
# =============================================================================

MIGRATION_SQL = """
-- Add organization_id column (required for multi-tenancy)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS organization_id VARCHAR(36);

-- Add folder_name column (optional folder context)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS folder_name VARCHAR(255);

-- Add status column with default 'uploaded'
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'uploaded' NOT NULL;

-- Add parsed_path column (path to parsed .md file)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parsed_path TEXT;

-- Add parsed_at column (parsing timestamp)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS parsed_at TIMESTAMP WITH TIME ZONE;

-- Add unique constraint on file_path if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'documents_file_path_key') THEN
        ALTER TABLE documents ADD CONSTRAINT documents_file_path_key UNIQUE (file_path);
    END IF;
END $$;

-- Add check constraint for status values
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_documents_status') THEN
        ALTER TABLE documents ADD CONSTRAINT chk_documents_status CHECK (status IN ('uploaded', 'parsed', 'failed'));
    END IF;
END $$;

-- Create indexes if they don't exist
CREATE INDEX IF NOT EXISTS idx_documents_file_path ON documents (file_path);
CREATE INDEX IF NOT EXISTS idx_documents_org_id ON documents (organization_id);
CREATE INDEX IF NOT EXISTS idx_documents_org_status ON documents (organization_id, status);
"""


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

async def get_connection(database: str):
    """Get a database connection."""
    instance_name = os.getenv("CLOUD_SQL_INSTANCE")
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_password = os.getenv("DATABASE_PASSWORD", "")
    use_connector = os.getenv("USE_CLOUD_SQL_CONNECTOR", "true").lower() == "true"
    ip_type_str = os.getenv("CLOUD_SQL_IP_TYPE", "PUBLIC").upper()

    conn = None
    connector = None

    if use_connector and instance_name:
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
# MIGRATION
# =============================================================================

async def run_migration(dry_run: bool = False):
    """Run the migration."""
    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")

    logger.info("=" * 60)
    logger.info("Document Status Migration")
    logger.info("=" * 60)
    logger.info(f"Target database: {target_db}")

    if dry_run:
        logger.info("\n[DRY RUN] SQL to be executed:\n")
        print(MIGRATION_SQL)
        return

    logger.info(f"\nConnecting to '{target_db}'...")
    conn, connector = await get_connection(target_db)

    try:
        # Check current table structure
        logger.info("Checking current table structure...")
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'documents'
            ORDER BY ordinal_position
        """)

        existing_columns = {col['column_name'] for col in columns}
        logger.info(f"Existing columns: {', '.join(existing_columns)}")

        # Check what needs to be added
        new_columns = {'organization_id', 'folder_name', 'status', 'parsed_path', 'parsed_at'}
        missing = new_columns - existing_columns

        if not missing:
            logger.info("All status tracking columns already exist!")
            return

        logger.info(f"Missing columns to add: {', '.join(missing)}")

        # Run migration
        logger.info("\nRunning migration...")
        await conn.execute(MIGRATION_SQL)

        # Verify migration
        logger.info("\nVerifying migration...")
        columns_after = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'documents'
            ORDER BY ordinal_position
        """)

        logger.info("Updated table structure:")
        for col in columns_after:
            logger.info(f"  - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")

        logger.info("\n" + "=" * 60)
        logger.info("Migration completed successfully!")
        logger.info("=" * 60)

    finally:
        await close_connection(conn, connector)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate documents table to add status tracking columns"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show SQL without executing"
    )

    args = parser.parse_args()

    asyncio.run(run_migration(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
