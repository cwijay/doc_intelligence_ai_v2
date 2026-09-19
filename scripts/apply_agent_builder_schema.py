#!/usr/bin/env python3
"""Apply the agent_builder schema migration to the GCP Cloud SQL instance.

Reads scripts/migrations/agent_builder_schema.sql and executes it against the
database selected by the current environment (respects USE_CLOUD_SQL_CONNECTOR
and CLOUD_SQL_INSTANCE, same way scripts/db_setup.py does).

Usage:
    # with .env.local-gcp loaded by start_local_gcp.sh --migrate
    python scripts/apply_agent_builder_schema.py

    # standalone, explicit env file
    set -a; source .env.local-gcp; set +a
    python scripts/apply_agent_builder_schema.py

    # dry run (print SQL, no execution)
    python scripts/apply_agent_builder_schema.py --dry-run

    # verify after apply
    python scripts/apply_agent_builder_schema.py --verify
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Allow imports from repo root so we can reuse db_setup helpers
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

# Load .env.local-gcp first if it exists (start_local_gcp.sh already exports,
# but running this standalone should still work).
for candidate in (".env.local-gcp", ".env_dev", ".env"):
    path = REPO_ROOT / candidate
    if path.exists():
        load_dotenv(path, override=False)
        break

from scripts.db_setup import get_connection, close_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agent_builder_migration")

MIGRATION_SQL_PATH = REPO_ROOT / "scripts" / "migrations" / "agent_builder_schema.sql"
EXPECTED_TABLES = ("agent_definitions", "agent_versions", "agent_runs")


def load_migration_sql() -> str:
    if not MIGRATION_SQL_PATH.exists():
        raise FileNotFoundError(f"Migration SQL not found: {MIGRATION_SQL_PATH}")
    return MIGRATION_SQL_PATH.read_text()


async def apply_migration() -> None:
    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    logger.info("Applying agent_builder migration to database '%s'", target_db)

    sql = load_migration_sql()
    conn, connector = await get_connection(target_db)
    try:
        await conn.execute(sql)
        logger.info("Migration applied successfully")
    finally:
        await close_connection(conn, connector)


async def verify_tables() -> int:
    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    logger.info("Verifying tables in '%s'", target_db)

    conn, connector = await get_connection(target_db)
    try:
        missing: list[str] = []
        for table in EXPECTED_TABLES:
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = $1
                )
                """,
                table,
            )
            if exists:
                count = await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')
                logger.info("  %-22s OK  (%d rows)", table, count)
            else:
                missing.append(table)
                logger.error("  %-22s MISSING", table)
        return 0 if not missing else 1
    finally:
        await close_connection(conn, connector)


def print_sql() -> None:
    print(load_migration_sql())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print SQL and exit")
    parser.add_argument("--verify", action="store_true", help="Only verify tables exist")
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Apply without running verification step",
    )
    args = parser.parse_args()

    if args.dry_run:
        print_sql()
        return 0

    if args.verify:
        return asyncio.run(verify_tables())

    asyncio.run(apply_migration())
    if args.skip_verify:
        return 0
    return asyncio.run(verify_tables())


if __name__ == "__main__":
    sys.exit(main())
