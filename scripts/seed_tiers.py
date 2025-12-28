#!/usr/bin/env python3
"""
Seed subscription tiers into the database.

DEPRECATED: This script has been moved to biz2bricks_infra.
Use `biz2bricks seed tiers` from the biz2bricks_infra CLI instead.

Creates default Free, Pro, and Enterprise tiers with configured limits.

Usage (deprecated - use biz2bricks CLI instead):
    python scripts/seed_tiers.py           # Seed default tiers
    python scripts/seed_tiers.py --list    # List current tiers
    python scripts/seed_tiers.py --reset   # Delete and re-seed tiers

Recommended (use biz2bricks_infra CLI):
    biz2bricks seed tiers           # Seed default tiers
    biz2bricks seed tiers --list    # List current tiers
    biz2bricks seed tiers --reset   # Delete and re-seed tiers
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from decimal import Decimal
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
# DEFAULT TIER CONFIGURATION
# =============================================================================

DEFAULT_TIERS = [
    {
        "tier": "free",
        "display_name": "Free",
        "description": "Free tier for individual users and small teams. Perfect for trying out Document Intelligence.",
        "monthly_token_limit": 50_000,
        "monthly_llamaparse_pages": 50,
        "monthly_file_search_queries": 100,
        "storage_gb_limit": Decimal("1.0"),
        "requests_per_minute": 10,
        "requests_per_day": 1000,
        "max_file_size_mb": 25,
        "max_concurrent_jobs": 2,
        "features": {
            "document_agent": True,
            "sheets_agent": True,
            "rag_search": True,
            "custom_models": False,
            "priority_support": False,
            "api_access": True,
        },
        "monthly_price_usd": Decimal("0.00"),
        "annual_price_usd": Decimal("0.00"),
        "sort_order": 1,
    },
    {
        "tier": "pro",
        "display_name": "Pro",
        "description": "Pro tier for growing teams. 10x the limits of Free with priority support.",
        "monthly_token_limit": 500_000,
        "monthly_llamaparse_pages": 500,
        "monthly_file_search_queries": 1000,
        "storage_gb_limit": Decimal("10.0"),
        "requests_per_minute": 60,
        "requests_per_day": 10000,
        "max_file_size_mb": 100,
        "max_concurrent_jobs": 10,
        "features": {
            "document_agent": True,
            "sheets_agent": True,
            "rag_search": True,
            "custom_models": True,
            "priority_support": True,
            "api_access": True,
            "advanced_analytics": True,
            "team_management": True,
        },
        "monthly_price_usd": Decimal("29.00"),
        "annual_price_usd": Decimal("290.00"),
        "sort_order": 2,
    },
    {
        "tier": "enterprise",
        "display_name": "Enterprise",
        "description": "Enterprise tier for large organizations. Custom limits available on request.",
        "monthly_token_limit": 5_000_000,
        "monthly_llamaparse_pages": 5000,
        "monthly_file_search_queries": 10000,
        "storage_gb_limit": Decimal("100.0"),
        "requests_per_minute": 300,
        "requests_per_day": 100000,
        "max_file_size_mb": 500,
        "max_concurrent_jobs": 50,
        "features": {
            "document_agent": True,
            "sheets_agent": True,
            "rag_search": True,
            "custom_models": True,
            "priority_support": True,
            "api_access": True,
            "advanced_analytics": True,
            "team_management": True,
            "sso": True,
            "audit_logs": True,
            "custom_integrations": True,
            "dedicated_support": True,
        },
        "monthly_price_usd": Decimal("199.00"),
        "annual_price_usd": Decimal("1990.00"),
        "sort_order": 3,
    },
]


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


async def list_tiers():
    """List existing subscription tiers."""
    conn, connector = await get_connection()
    try:
        rows = await conn.fetch("""
            SELECT id, tier, display_name, monthly_token_limit,
                   monthly_llamaparse_pages, monthly_file_search_queries,
                   storage_gb_limit, monthly_price_usd, is_active
            FROM subscription_tiers
            ORDER BY sort_order
        """)

        if not rows:
            logger.info("No subscription tiers found in database")
            return

        logger.info(f"\nSubscription Tiers ({len(rows)}):")
        logger.info("-" * 80)

        for row in rows:
            status = "active" if row['is_active'] else "inactive"
            logger.info(
                f"  {row['tier']:12} | "
                f"Tokens: {row['monthly_token_limit']:>10,} | "
                f"Pages: {row['monthly_llamaparse_pages']:>5} | "
                f"Queries: {row['monthly_file_search_queries']:>6} | "
                f"Storage: {row['storage_gb_limit']:>5} GB | "
                f"${row['monthly_price_usd']}/mo | "
                f"[{status}]"
            )

    finally:
        await close_connection(conn, connector)


async def delete_all_tiers():
    """Delete all subscription tiers."""
    conn, connector = await get_connection()
    try:
        # First check for subscriptions using these tiers
        sub_count = await conn.fetchval(
            "SELECT COUNT(*) FROM organization_subscriptions"
        )

        if sub_count > 0:
            logger.warning(
                f"Warning: {sub_count} organization subscriptions exist. "
                "Deleting tiers may affect these subscriptions."
            )
            confirm = input("Continue? (yes/no): ")
            if confirm.lower() != "yes":
                logger.info("Operation cancelled")
                return False

        await conn.execute("DELETE FROM subscription_tiers")
        logger.info("Deleted all subscription tiers")
        return True

    finally:
        await close_connection(conn, connector)


async def seed_tiers():
    """Seed default subscription tiers."""
    conn, connector = await get_connection()
    try:
        now = datetime.utcnow()

        for tier_config in DEFAULT_TIERS:
            tier_id = uuid.uuid4()

            # Check if tier already exists
            existing = await conn.fetchval(
                "SELECT id FROM subscription_tiers WHERE tier = $1",
                tier_config["tier"]
            )

            if existing:
                # Update existing tier
                await conn.execute(
                    """
                    UPDATE subscription_tiers SET
                        display_name = $2,
                        description = $3,
                        monthly_token_limit = $4,
                        monthly_llamaparse_pages = $5,
                        monthly_file_search_queries = $6,
                        storage_gb_limit = $7,
                        requests_per_minute = $8,
                        requests_per_day = $9,
                        max_file_size_mb = $10,
                        max_concurrent_jobs = $11,
                        features = $12,
                        monthly_price_usd = $13,
                        annual_price_usd = $14,
                        sort_order = $15,
                        updated_at = $16
                    WHERE tier = $1
                    """,
                    tier_config["tier"],
                    tier_config["display_name"],
                    tier_config["description"],
                    tier_config["monthly_token_limit"],
                    tier_config["monthly_llamaparse_pages"],
                    tier_config["monthly_file_search_queries"],
                    tier_config["storage_gb_limit"],
                    tier_config["requests_per_minute"],
                    tier_config["requests_per_day"],
                    tier_config["max_file_size_mb"],
                    tier_config["max_concurrent_jobs"],
                    json.dumps(tier_config["features"]),  # JSONB needs JSON string
                    tier_config["monthly_price_usd"],
                    tier_config["annual_price_usd"],
                    tier_config["sort_order"],
                    now,
                )
                logger.info(f"Updated tier: {tier_config['tier']}")
            else:
                # Insert new tier
                await conn.execute(
                    """
                    INSERT INTO subscription_tiers (
                        id, tier, display_name, description,
                        monthly_token_limit, monthly_llamaparse_pages,
                        monthly_file_search_queries, storage_gb_limit,
                        requests_per_minute, requests_per_day,
                        max_file_size_mb, max_concurrent_jobs,
                        features, monthly_price_usd, annual_price_usd,
                        is_active, sort_order, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                              $11, $12, $13, $14, $15, $16, $17, $18, $19)
                    """,
                    tier_id,
                    tier_config["tier"],
                    tier_config["display_name"],
                    tier_config["description"],
                    tier_config["monthly_token_limit"],
                    tier_config["monthly_llamaparse_pages"],
                    tier_config["monthly_file_search_queries"],
                    tier_config["storage_gb_limit"],
                    tier_config["requests_per_minute"],
                    tier_config["requests_per_day"],
                    tier_config["max_file_size_mb"],
                    tier_config["max_concurrent_jobs"],
                    json.dumps(tier_config["features"]),  # JSONB needs JSON string
                    tier_config["monthly_price_usd"],
                    tier_config["annual_price_usd"],
                    True,  # is_active
                    tier_config["sort_order"],
                    now,
                    now,
                )
                logger.info(f"Created tier: {tier_config['tier']}")

        logger.info("\nSuccessfully seeded subscription tiers!")

    finally:
        await close_connection(conn, connector)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Seed subscription tiers into the database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/seed_tiers.py           # Seed default tiers (upsert)
    python scripts/seed_tiers.py --list    # List current tiers
    python scripts/seed_tiers.py --reset   # Delete and re-seed tiers

Default Tiers:
    Free       - 50,000 tokens/mo, 50 pages, 100 queries, 1 GB storage
    Pro        - 500,000 tokens/mo, 500 pages, 1,000 queries, 10 GB storage ($29/mo)
    Enterprise - 5,000,000 tokens/mo, 5,000 pages, 10,000 queries, 100 GB storage ($199/mo)
        """
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List current subscription tiers"
    )
    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="Delete all tiers and re-seed"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Document Intelligence AI - Subscription Tier Seeder")
    print("=" * 60 + "\n")

    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    print(f"Target database: {target_db}\n")

    if args.list:
        asyncio.run(list_tiers())
    elif args.reset:
        asyncio.run(delete_all_tiers())
        asyncio.run(seed_tiers())
        asyncio.run(list_tiers())
    else:
        asyncio.run(seed_tiers())
        asyncio.run(list_tiers())


if __name__ == "__main__":
    main()
