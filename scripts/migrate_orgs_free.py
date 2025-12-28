#!/usr/bin/env python3
"""
Migrate existing organizations to Free tier.

Creates organization_subscriptions records for all organizations
that don't already have one, assigning them to the Free tier.

Usage:
    python scripts/migrate_orgs_free.py              # Dry run - show what would be done
    python scripts/migrate_orgs_free.py --execute    # Actually perform the migration
    python scripts/migrate_orgs_free.py --status     # Show migration status
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
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
# DATABASE OPERATIONS
# =============================================================================

async def get_connection():
    """Get database connection."""
    instance_name = os.getenv(
        "CLOUD_SQL_INSTANCE",
        "biz2bricks-dev-v1:us-central1:biz-2-bricks-intelli-doc-dev"
    )
    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    db_user = os.getenv("DATABASE_USER", "postgres")
    db_password = os.getenv("DATABASE_PASSWORD", "")
    use_connector = os.getenv("USE_CLOUD_SQL_CONNECTOR", "true").lower() == "true"
    ip_type_str = os.getenv("CLOUD_SQL_IP_TYPE", "PUBLIC").upper()

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
                db=target_db,
                ip_type=ip_type,
            )
            return conn, connector
        except Exception as e:
            logger.warning(f"Cloud SQL Connector failed: {e}, falling back to direct connection")

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
        database=target_db,
    )
    return conn, None


async def close_connection(conn, connector):
    """Close database connection."""
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


async def get_migration_status():
    """Get current migration status."""
    conn, connector = await get_connection()
    try:
        # Count total organizations
        org_count = await conn.fetchval("SELECT COUNT(*) FROM organizations")

        # Count organizations with subscriptions
        sub_count = await conn.fetchval("SELECT COUNT(*) FROM organization_subscriptions")

        # Count organizations without subscriptions
        missing_count = await conn.fetchval("""
            SELECT COUNT(*)
            FROM organizations o
            LEFT JOIN organization_subscriptions os ON o.id = os.organization_id
            WHERE os.id IS NULL
        """)

        # Get tier distribution
        tier_dist = await conn.fetch("""
            SELECT st.tier, COUNT(*) as count
            FROM organization_subscriptions os
            JOIN subscription_tiers st ON os.tier_id = st.id
            GROUP BY st.tier
            ORDER BY st.sort_order
        """)

        logger.info("\n" + "=" * 60)
        logger.info("Migration Status")
        logger.info("=" * 60)
        logger.info(f"\nTotal organizations:          {org_count}")
        logger.info(f"Organizations with subscription: {sub_count}")
        logger.info(f"Organizations without subscription: {missing_count}")

        if tier_dist:
            logger.info("\nTier distribution:")
            for row in tier_dist:
                logger.info(f"  {row['tier']:12}: {row['count']}")

        return {
            "org_count": org_count,
            "sub_count": sub_count,
            "missing_count": missing_count,
        }

    finally:
        await close_connection(conn, connector)


async def get_orgs_without_subscription():
    """Get list of organizations without subscriptions."""
    conn, connector = await get_connection()
    try:
        rows = await conn.fetch("""
            SELECT o.id, o.name
            FROM organizations o
            LEFT JOIN organization_subscriptions os ON o.id = os.organization_id
            WHERE os.id IS NULL
            ORDER BY o.name
        """)

        return [(row['id'], row['name']) for row in rows]

    finally:
        await close_connection(conn, connector)


async def migrate_orgs_to_free(dry_run: bool = True):
    """
    Migrate organizations without subscriptions to Free tier.

    Args:
        dry_run: If True, only show what would be done without making changes
    """
    conn, connector = await get_connection()
    try:
        # Get Free tier
        free_tier = await conn.fetchrow(
            "SELECT * FROM subscription_tiers WHERE tier = 'free' AND is_active = true"
        )

        if not free_tier:
            logger.error("Free tier not found in subscription_tiers table!")
            logger.error("Please run 'python scripts/seed_tiers.py' first.")
            return

        # Get orgs without subscription
        orgs = await conn.fetch("""
            SELECT o.id, o.name
            FROM organizations o
            LEFT JOIN organization_subscriptions os ON o.id = os.organization_id
            WHERE os.id IS NULL
            ORDER BY o.name
        """)

        if not orgs:
            logger.info("\nNo organizations need migration. All organizations have subscriptions.")
            return

        logger.info(f"\nFound {len(orgs)} organizations without subscriptions:")

        for org in orgs[:10]:  # Show first 10
            logger.info(f"  - {org['name']} ({org['id']})")
        if len(orgs) > 10:
            logger.info(f"  ... and {len(orgs) - 10} more")

        if dry_run:
            logger.info("\n[DRY RUN] No changes will be made.")
            logger.info("Run with --execute to perform the migration.")
            return

        # Perform migration
        logger.info("\nMigrating organizations to Free tier...")

        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        period_end = (period_start + timedelta(days=32)).replace(day=1)

        # Calculate storage limit in bytes
        storage_limit_bytes = int(float(free_tier['storage_gb_limit']) * 1024 * 1024 * 1024)

        migrated = 0
        for org in orgs:
            sub_id = uuid.uuid4()
            try:
                await conn.execute(
                    """
                    INSERT INTO organization_subscriptions (
                        id, organization_id, tier_id, status, billing_cycle,
                        current_period_start, current_period_end,
                        tokens_used_this_period, llamaparse_pages_used,
                        file_search_queries_used, storage_used_bytes,
                        monthly_token_limit, monthly_llamaparse_pages_limit,
                        monthly_file_search_queries_limit, storage_limit_bytes,
                        created_at, updated_at
                    ) VALUES ($1, $2, $3, 'active', 'monthly',
                              $4, $5, 0, 0, 0, 0,
                              $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (organization_id) DO NOTHING
                    """,
                    sub_id,
                    org['id'],
                    free_tier['id'],
                    period_start,
                    period_end,
                    free_tier['monthly_token_limit'],
                    free_tier['monthly_llamaparse_pages'],
                    free_tier['monthly_file_search_queries'],
                    storage_limit_bytes,
                    now,
                    now,
                )
                migrated += 1
            except Exception as e:
                logger.error(f"Failed to migrate org {org['id']}: {e}")

        logger.info(f"\nSuccessfully migrated {migrated}/{len(orgs)} organizations to Free tier!")

    finally:
        await close_connection(conn, connector)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing organizations to Free tier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates organization_subscriptions records for all organizations
that don't already have one, assigning them to the Free tier.

The Free tier provides:
    - 50,000 tokens/month
    - 50 LlamaParse pages/month
    - 100 file search queries/month
    - 1 GB storage

Examples:
    python scripts/migrate_orgs_free.py              # Dry run
    python scripts/migrate_orgs_free.py --execute    # Execute migration
    python scripts/migrate_orgs_free.py --status     # Show status only
        """
    )

    parser.add_argument(
        "--execute", "-e",
        action="store_true",
        help="Actually perform the migration (default is dry run)"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show migration status only"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Document Intelligence AI - Organization Migration")
    print("=" * 60 + "\n")

    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    print(f"Target database: {target_db}\n")

    if args.status:
        asyncio.run(get_migration_status())
    else:
        asyncio.run(get_migration_status())
        asyncio.run(migrate_orgs_to_free(dry_run=not args.execute))

        if not args.execute:
            print("\n" + "-" * 60)
            print("To execute the migration, run:")
            print("  python scripts/migrate_orgs_free.py --execute")


if __name__ == "__main__":
    main()
