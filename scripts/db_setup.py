#!/usr/bin/env python3
"""
Database setup script for Document Intelligence AI.

Uses SQLAlchemy models as the SINGLE SOURCE OF TRUTH for schema.
All tables, indexes, and constraints are defined in src/db/models.py.

Usage:
    python scripts/db_setup.py setup      # Create database and all tables
    python scripts/db_setup.py teardown   # Drop all tables (with confirmation)
    python scripts/db_setup.py reset      # Teardown + setup (full reset)
    python scripts/db_setup.py status     # Show current database state

Environment variables (from .env):
    - CLOUD_SQL_INSTANCE: Cloud SQL instance connection name
    - DATABASE_NAME: Target database name (default: doc_intelligence)
    - DATABASE_USER: PostgreSQL username (default: postgres)
    - DATABASE_PASSWORD: PostgreSQL password
    - USE_CLOUD_SQL_CONNECTOR: Use Cloud SQL connector (default: true)
    - CLOUD_SQL_IP_TYPE: PUBLIC or PRIVATE (default: PUBLIC)
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
# DATABASE CONNECTION HELPERS
# =============================================================================

async def get_connection(database: str = "postgres"):
    """
    Get a database connection using Cloud SQL Connector or direct connection.

    Args:
        database: Database name to connect to (default: postgres for admin operations)

    Returns:
        tuple: (connection, connector) - connector is None for direct connections
    """
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


def get_sync_engine(database: str):
    """
    Get a synchronous SQLAlchemy engine for schema operations.

    Uses Cloud SQL Connector if available, otherwise direct connection.
    """
    from sqlalchemy import create_engine

    db_user = os.getenv("DATABASE_USER", "postgres")
    db_password = os.getenv("DATABASE_PASSWORD", "")
    use_connector = os.getenv("USE_CLOUD_SQL_CONNECTOR", "true").lower() == "true"
    instance_name = os.getenv("CLOUD_SQL_INSTANCE", "")
    ip_type_str = os.getenv("CLOUD_SQL_IP_TYPE", "PUBLIC").upper()

    if use_connector and instance_name:
        try:
            from google.cloud.sql.connector import Connector, IPTypes

            ip_type = IPTypes.PUBLIC if ip_type_str == "PUBLIC" else IPTypes.PRIVATE
            connector = Connector()

            def getconn():
                return connector.connect(
                    instance_name,
                    "pg8000",
                    user=db_user,
                    password=db_password,
                    db=database,
                    ip_type=ip_type,
                )

            engine = create_engine(
                "postgresql+pg8000://",
                creator=getconn,
            )
            return engine, connector
        except Exception as e:
            logger.warning(f"Cloud SQL Connector failed: {e}, falling back to direct connection")

    # Direct connection fallback
    db_url = os.getenv("DATABASE_URL", "")
    if db_url:
        # Replace async driver with sync driver
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
        # Replace database name if different
        if f"/{database}" not in sync_url:
            sync_url = sync_url.rsplit("/", 1)[0] + f"/{database}"
    else:
        host = "localhost"
        port = "5432"
        sync_url = f"postgresql://{db_user}:{db_password}@{host}:{port}/{database}"

    engine = create_engine(sync_url)
    return engine, None


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

async def create_database():
    """Create the target database if it doesn't exist."""
    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")

    logger.info(f"Checking if database '{target_db}' exists...")

    conn, connector = await get_connection("postgres")
    try:
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_db
        )

        if result:
            logger.info(f"Database '{target_db}' already exists")
            return True

        logger.info(f"Creating database '{target_db}'...")
        await conn.execute(f'CREATE DATABASE "{target_db}"')
        logger.info(f"Database '{target_db}' created successfully")
        return True

    finally:
        await close_connection(conn, connector)


async def enable_extensions():
    """Enable required PostgreSQL extensions (pgvector for semantic caching)."""
    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")

    logger.info("Enabling PostgreSQL extensions...")

    conn, connector = await get_connection(target_db)
    try:
        # Enable pgvector extension for semantic similarity search
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("Enabled extension: vector (pgvector)")

    except Exception as e:
        logger.warning(f"Could not enable pgvector extension: {e}")
        logger.warning("Semantic caching will not be available. Install pgvector on your PostgreSQL server.")

    finally:
        await close_connection(conn, connector)


async def create_vector_indexes():
    """Create vector indexes for semantic similarity search (requires pgvector)."""
    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")

    conn, connector = await get_connection(target_db)
    try:
        # Check if rag_query_cache table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'rag_query_cache'
            )
        """)

        if not table_exists:
            logger.info("Skipping vector index creation: rag_query_cache table does not exist yet")
            return

        # Check if vector extension is available
        ext_exists = await conn.fetchval("""
            SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')
        """)

        if not ext_exists:
            logger.warning("Skipping vector index: pgvector extension not available")
            return

        # Create IVFFlat index for fast approximate nearest neighbor search
        # Using cosine similarity (vector_cosine_ops)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_rag_cache_embedding
            ON rag_query_cache
            USING ivfflat (query_embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
        logger.info("Created vector index: idx_rag_cache_embedding")

    except Exception as e:
        logger.warning(f"Could not create vector index: {e}")

    finally:
        await close_connection(conn, connector)


def create_tables_from_models():
    """
    Create all tables using SQLAlchemy ORM models.

    This is the primary method - models.py is the single source of truth.
    """
    from src.db.models import Base

    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    logger.info(f"Creating tables from SQLAlchemy models...")

    engine, connector = get_sync_engine(target_db)
    try:
        # Create all tables defined in models.py
        Base.metadata.create_all(engine)

        # Log created tables
        table_names = list(Base.metadata.tables.keys())
        logger.info(f"Created {len(table_names)} tables: {', '.join(sorted(table_names))}")

    finally:
        engine.dispose()
        if connector:
            connector.close()


def drop_tables_from_models():
    """
    Drop all tables using SQLAlchemy ORM models.
    """
    from src.db.models import Base

    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    logger.info(f"Dropping tables from SQLAlchemy models...")

    engine, connector = get_sync_engine(target_db)
    try:
        # Drop all tables defined in models.py
        Base.metadata.drop_all(engine)

        table_names = list(Base.metadata.tables.keys())
        logger.info(f"Dropped {len(table_names)} tables")

    finally:
        engine.dispose()
        if connector:
            connector.close()


# =============================================================================
# CLI COMMANDS
# =============================================================================

async def cmd_setup():
    """Setup command: Create database and tables."""
    logger.info("=" * 60)
    logger.info("Database Setup")
    logger.info("=" * 60)

    # Create database first (requires async connection)
    await create_database()

    # Enable PostgreSQL extensions (pgvector for semantic caching)
    await enable_extensions()

    # Create tables using SQLAlchemy models
    create_tables_from_models()

    # Create vector indexes for semantic similarity search
    await create_vector_indexes()

    logger.info("=" * 60)
    logger.info("Setup completed successfully!")
    logger.info("=" * 60)


async def cmd_teardown(force: bool = False):
    """Teardown command: Drop all tables."""
    logger.info("=" * 60)
    logger.info("Database Teardown")
    logger.info("=" * 60)

    if not force:
        confirm = input("Are you sure you want to DROP all tables? This cannot be undone. (yes/no): ")
        if confirm.lower() != "yes":
            logger.info("Operation cancelled")
            return

    drop_tables_from_models()

    logger.info("=" * 60)
    logger.info("Teardown completed successfully!")
    logger.info("=" * 60)


async def cmd_reset(force: bool = False):
    """Reset command: Teardown + Setup."""
    logger.info("=" * 60)
    logger.info("Database Reset")
    logger.info("=" * 60)

    if not force:
        confirm = input("Are you sure you want to RESET the database? All data will be lost. (yes/no): ")
        if confirm.lower() != "yes":
            logger.info("Operation cancelled")
            return

    # Drop tables
    try:
        drop_tables_from_models()
    except Exception as e:
        logger.warning(f"Teardown error (may be expected if tables don't exist): {e}")

    # Create database and tables
    await create_database()
    await enable_extensions()
    create_tables_from_models()
    await create_vector_indexes()

    logger.info("=" * 60)
    logger.info("Reset completed successfully!")
    logger.info("=" * 60)


async def cmd_status():
    """Status command: Show current database state."""
    logger.info("=" * 60)
    logger.info("Database Status")
    logger.info("=" * 60)

    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")

    # Check if database exists
    conn, connector = await get_connection("postgres")
    try:
        result = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_db
        )

        if not result:
            logger.info(f"Database '{target_db}' does NOT exist")
            return

        logger.info(f"Database '{target_db}' exists")
    finally:
        await close_connection(conn, connector)

    # Connect to target database and check tables
    conn, connector = await get_connection(target_db)
    try:
        # Get existing tables
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)

        logger.info(f"\nTables ({len(tables)}):")
        for table in tables:
            table_name = table['table_name']
            count = await conn.fetchval(f'SELECT COUNT(*) FROM "{table_name}"')
            logger.info(f"  - {table_name}: {count} rows")

        # Check for missing tables from models
        from src.db.models import Base
        expected_tables = set(Base.metadata.tables.keys())
        existing_tables = {t['table_name'] for t in tables}
        missing = expected_tables - existing_tables
        if missing:
            logger.info(f"\nMissing tables (defined in models but not in DB): {', '.join(sorted(missing))}")

        extra = existing_tables - expected_tables
        if extra:
            logger.info(f"\nExtra tables (in DB but not in models): {', '.join(sorted(extra))}")

        # Get indexes
        indexes = await conn.fetch("""
            SELECT indexname, tablename
            FROM pg_indexes
            WHERE schemaname = 'public'
            AND indexname NOT LIKE '%_pkey'
            ORDER BY tablename, indexname
        """)

        logger.info(f"\nCustom indexes ({len(indexes)}):")
        for idx in indexes:
            logger.info(f"  - {idx['indexname']} on {idx['tablename']}")

    finally:
        await close_connection(conn, connector)


def cmd_models():
    """Models command: List all models and their tables/columns."""
    from src.db.models import Base

    print("=" * 60)
    print("SQLAlchemy Models (Single Source of Truth)")
    print("=" * 60)
    print()

    for table_name, table in sorted(Base.metadata.tables.items()):
        print(f"Table: {table_name}")
        print("-" * 40)

        # Columns
        for column in table.columns:
            nullable = "NULL" if column.nullable else "NOT NULL"
            pk = " PRIMARY KEY" if column.primary_key else ""
            fk = ""
            if column.foreign_keys:
                fk_refs = [str(fk.target_fullname) for fk in column.foreign_keys]
                fk = f" -> {', '.join(fk_refs)}"
            print(f"  {column.name}: {column.type} {nullable}{pk}{fk}")

        # Indexes
        for index in table.indexes:
            cols = ", ".join([c.name for c in index.columns])
            print(f"  INDEX {index.name} ({cols})")

        # Constraints
        for constraint in table.constraints:
            if hasattr(constraint, 'name') and constraint.name and not constraint.name.endswith('_pkey'):
                print(f"  CONSTRAINT {constraint.name}")

        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Database setup script for Document Intelligence AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  setup      Create database and all tables from models
  teardown   Drop all tables (with confirmation)
  reset      Teardown + setup (full reset)
  status     Show current database state
  models     List all SQLAlchemy models and their schema

Examples:
  python scripts/db_setup.py setup              # Create all tables
  python scripts/db_setup.py status             # Show database state
  python scripts/db_setup.py teardown --force   # Drop tables without confirmation
  python scripts/db_setup.py reset --force      # Reset database
  python scripts/db_setup.py models             # Show model definitions

Schema is defined in: src/db/models.py (SINGLE SOURCE OF TRUTH)
        """
    )

    parser.add_argument(
        "command",
        choices=["setup", "teardown", "reset", "status", "models"],
        help="Command to execute"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Skip confirmation prompts for destructive operations"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    print("\n" + "=" * 60)
    print("Document Intelligence AI - Database Setup")
    print("=" * 60 + "\n")

    target_db = os.getenv("DATABASE_NAME", "doc_intelligence")
    print(f"Target database: {target_db}")
    print(f"Cloud SQL instance: {os.getenv('CLOUD_SQL_INSTANCE', 'localhost')}")
    print(f"Schema source: src/db/models.py")
    print()

    # Execute command
    if args.command == "models":
        cmd_models()
    elif args.command == "setup":
        asyncio.run(cmd_setup())
    elif args.command == "teardown":
        asyncio.run(cmd_teardown(force=args.force))
    elif args.command == "reset":
        asyncio.run(cmd_reset(force=args.force))
    elif args.command == "status":
        asyncio.run(cmd_status())


if __name__ == "__main__":
    main()
