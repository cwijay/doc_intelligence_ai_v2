"""
PostgreSQL async connection management using SQLAlchemy 2.0.

Supports both:
- Google Cloud SQL Python Connector (for production)
- Direct connection URL (for local development)

Note: Uses per-event-loop connector management to handle multi-threaded
async operations (e.g., background audit logging from ThreadPoolExecutor).
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import AsyncAdaptedQueuePool

from src.utils.env_utils import parse_bool_env, parse_int_env

load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        # Database enabled flag - set to false to skip all DB operations
        self.enabled = parse_bool_env("DATABASE_ENABLED", True)

        # Cloud SQL settings - match .env variable names
        self.instance_connection_name = os.getenv(
            "CLOUD_SQL_INSTANCE",
            "biz2bricks-dev-v1:us-central1:biz-2-bricks-intelli-doc-dev"
        )
        self.db_name = os.getenv("DATABASE_NAME", "doc_intelligence")
        self.db_user = os.getenv("DATABASE_USER", "postgres")
        self.db_password = os.getenv("DATABASE_PASSWORD", "")
        # Cloud SQL IP type: "PUBLIC" or "PRIVATE" (default: PRIVATE for production)
        self.cloud_sql_ip_type = os.getenv("CLOUD_SQL_IP_TYPE", "PRIVATE").upper()

        # Connection pool settings
        # Main loop gets larger pool for API concurrency
        self.pool_size = parse_int_env("DB_POOL_SIZE", 3)
        # Background/executor loops get smaller pool (sequential operations)
        self.background_pool_size = parse_int_env("DB_BACKGROUND_POOL_SIZE", 2)
        self.max_overflow = parse_int_env("DB_MAX_OVERFLOW", 5)
        self.pool_timeout = parse_int_env("DB_POOL_TIMEOUT", 30)
        self.pool_recycle = parse_int_env("DB_POOL_RECYCLE", 1800)  # 30 min

        # Use Cloud SQL connector or direct connection
        self.use_cloud_sql_connector = parse_bool_env("USE_CLOUD_SQL_CONNECTOR", True)

        # Direct connection URL (for local development or testing)
        self.database_url = os.getenv(
            "DATABASE_URL",
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}@localhost:5432/{self.db_name}"
        )

        # Debug mode
        self.echo_sql = parse_bool_env("DB_ECHO", False)


class DatabaseManager:
    """
    Manages async PostgreSQL connections with Cloud SQL connector support.

    Implements singleton pattern with PER-LOOP engines to avoid
    "Future attached to different loop" errors when using background threads.
    Each event loop gets its own engine, connector, and connection pool.

    Pool sizes are differentiated:
    - Main loop: pool_size (default 3) for API request concurrency
    - Background loops: background_pool_size (default 2) for sequential operations
    """

    _instance: Optional["DatabaseManager"] = None
    _initialized: bool = False
    _shutdown: bool = False  # Prevents new connections after close_all()

    # Track the main event loop for pool size differentiation
    _main_loop_id: Optional[int] = None

    # Per-loop resources (each loop gets its own engine and pool)
    _connectors: Dict[int, Any] = {}  # Per-loop Cloud SQL connectors
    _engines: Dict[int, AsyncEngine] = {}  # Per-loop engines
    _session_factories: Dict[int, async_sessionmaker] = {}  # Per-loop session factories

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return  # Already initialized

        self.config = DatabaseConfig()
        self._initialized = True
        # Note: Engine setup is deferred until first use per event loop

    def _get_loop_id(self) -> int:
        """
        Get current event loop ID for per-loop resource tracking.

        Returns the id of the running event loop, or 0 if no loop is running.
        """
        try:
            loop = asyncio.get_running_loop()
            return id(loop)
        except RuntimeError:
            # No running loop - use a default key
            return 0

    def _get_pool_size_for_loop(self, loop_id: int) -> int:
        """
        Get appropriate pool size for a given event loop.

        Main loop gets larger pool for API concurrency.
        Background/executor loops get smaller pool for sequential operations.
        """
        # First engine created is considered the main loop
        if self._main_loop_id is None:
            self._main_loop_id = loop_id
            return self.config.pool_size  # Main loop: default 3

        if loop_id == self._main_loop_id:
            return self.config.pool_size  # Main loop: default 3

        return self.config.background_pool_size  # Background: default 2

    async def _async_setup_engine_for_loop(self, loop_id: int):
        """
        Initialize a NEW engine for this specific event loop.

        Each event loop gets its own engine with its own connection pool
        to avoid "Future attached to different loop" errors.
        """
        # Don't create new connections if shutting down
        if self._shutdown:
            logger.debug(f"Skipping engine setup for loop {loop_id} - shutdown in progress")
            return

        # Skip if database is disabled
        if not self.config.enabled:
            logger.debug(f"Skipping engine setup for loop {loop_id} - database disabled")
            return

        # Fast path: engine already exists for this loop
        if loop_id in self._engines:
            return

        # Get appropriate pool size for this loop
        pool_size = self._get_pool_size_for_loop(loop_id)
        is_main = loop_id == self._main_loop_id

        # Create engine for this loop
        if self.config.use_cloud_sql_connector:
            engine = await self._create_cloud_sql_engine_for_loop(loop_id, pool_size)
        else:
            engine = self._create_direct_engine(pool_size)

        self._engines[loop_id] = engine

        # Create session factory for this loop using its own engine
        self._session_factories[loop_id] = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

        loop_type = "main" if is_main else "background"
        logger.info(f"Created database engine for loop {loop_id} ({loop_type}, pool_size={pool_size})")

    async def _create_cloud_sql_engine_for_loop(self, loop_id: int, pool_size: int) -> AsyncEngine:
        """
        Create a dedicated engine with Cloud SQL connector for a specific event loop.

        Each loop gets its own connector and engine to avoid cross-loop Future issues.

        Args:
            loop_id: The event loop ID
            pool_size: Connection pool size for this engine
        """
        CLOUD_SQL_CONNECT_TIMEOUT = 30.0

        try:
            from google.cloud.sql.connector import Connector, IPTypes

            ip_type = IPTypes.PUBLIC if self.config.cloud_sql_ip_type == "PUBLIC" else IPTypes.PRIVATE
            loop = asyncio.get_running_loop()

            # Create connector bound to this specific loop
            connector = Connector(loop=loop)
            self._connectors[loop_id] = connector

            # Test connection
            logger.info(
                f"Creating Cloud SQL engine for loop {loop_id}: "
                f"instance={self.config.instance_connection_name}, "
                f"ip_type={self.config.cloud_sql_ip_type}"
            )

            try:
                test_conn = await asyncio.wait_for(
                    connector.connect_async(
                        self.config.instance_connection_name,
                        "asyncpg",
                        user=self.config.db_user,
                        password=self.config.db_password,
                        db=self.config.db_name,
                        ip_type=ip_type,
                    ),
                    timeout=CLOUD_SQL_CONNECT_TIMEOUT
                )
                await test_conn.close()
                logger.info(f"Cloud SQL connection test successful for loop {loop_id}")
            except asyncio.TimeoutError:
                logger.warning(
                    f"Cloud SQL Connector timed out for loop {loop_id}. "
                    f"Falling back to direct connection."
                )
                connector.close()
                del self._connectors[loop_id]
                return self._create_direct_engine(pool_size)
            except Exception as e:
                logger.warning(
                    f"Cloud SQL Connector failed for loop {loop_id} ({type(e).__name__}: {e}). "
                    f"Falling back to direct connection."
                )
                connector.close()
                del self._connectors[loop_id]
                return self._create_direct_engine(pool_size)

            # Capture config for the getconn closure
            instance_name = self.config.instance_connection_name
            db_user = self.config.db_user
            db_password = self.config.db_password
            db_name = self.config.db_name
            this_connector = connector  # Use THIS loop's connector only

            async def getconn():
                """Connection factory using this loop's dedicated connector."""
                conn = await asyncio.wait_for(
                    this_connector.connect_async(
                        instance_name,
                        "asyncpg",
                        user=db_user,
                        password=db_password,
                        db=db_name,
                        ip_type=ip_type,
                    ),
                    timeout=CLOUD_SQL_CONNECT_TIMEOUT
                )
                return conn

            engine = create_async_engine(
                "postgresql+asyncpg://",
                async_creator=getconn,
                poolclass=AsyncAdaptedQueuePool,
                pool_size=pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo_sql,
            )

            return engine

        except ImportError as e:
            logger.warning(
                f"Cloud SQL connector not available ({e}), falling back to direct connection"
            )
            return self._create_direct_engine(pool_size)

    def _create_direct_engine(self, pool_size: int) -> AsyncEngine:
        """Create engine with direct connection URL.

        Args:
            pool_size: Connection pool size for this engine
        """
        # Mask password in log (show first 2 chars only)
        safe_url = self.config.database_url
        if "@" in safe_url and ":" in safe_url:
            # postgresql+asyncpg://user:pass@host:port/db -> mask the password
            try:
                parts = safe_url.split("@")
                credentials = parts[0].split(":")
                if len(credentials) >= 3:
                    safe_url = f"{credentials[0]}:{credentials[1]}:****@{parts[1]}"
            except Exception:
                safe_url = "[URL masked]"

        logger.info(f"Creating direct database connection: {safe_url} (pool_size={pool_size})")
        return create_async_engine(
            self.config.database_url,
            poolclass=AsyncAdaptedQueuePool,
            pool_size=pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo_sql,
        )

    @property
    def engine(self) -> AsyncEngine:
        """
        Get the async engine for the current event loop.

        NOTE: This property assumes the engine has already been initialized
        via session(). For first-time access, use get_engine_async() instead.
        """
        loop_id = self._get_loop_id()
        if loop_id not in self._engines:
            raise RuntimeError(
                "Engine not initialized for this event loop. "
                "Use 'async with db.session()' or 'await db.get_engine_async()' first."
            )
        return self._engines[loop_id]

    async def get_engine_async(self) -> Optional[AsyncEngine]:
        """Get the async engine, initializing for the current event loop if necessary."""
        if not self.config.enabled:
            return None
        loop_id = self._get_loop_id()
        if loop_id not in self._engines:
            await self._async_setup_engine_for_loop(loop_id)
        return self._engines.get(loop_id)

    async def test_connection(self, timeout: float = 15.0) -> bool:
        """
        Test database connectivity with timeout.

        Use this at startup to verify the database is reachable before
        accepting requests.

        Args:
            timeout: Maximum time to wait for connection test (seconds)

        Returns:
            True if connection successful, False otherwise
        """
        from sqlalchemy import text

        if not self.config.enabled:
            logger.info("Database disabled - skipping connection test")
            return True  # Skip if disabled

        engine = await self.get_engine_async()
        if not engine:
            logger.warning("No database engine available")
            return False

        try:
            async with asyncio.timeout(timeout):
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except asyncio.TimeoutError:
            logger.error(f"Database connection test timed out after {timeout}s")
            return False
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Optional[AsyncSession], None]:
        """
        Get an async session with automatic commit/rollback.

        Creates engine/connector for the current event loop if needed.
        Returns None if database is disabled.

        Usage:
            async with db.session() as session:
                if session:
                    result = await session.execute(...)

        The session will:
        - Commit on successful exit
        - Rollback on exception
        - Always close after use
        """
        # Return None if database is disabled
        if not self.config.enabled:
            yield None
            return

        loop_id = self._get_loop_id()
        if loop_id not in self._session_factories:
            await self._async_setup_engine_for_loop(loop_id)

        if loop_id not in self._session_factories:
            yield None
            return

        session = self._session_factories[loop_id]()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_tables(self):
        """
        Create all tables (for development/testing).

        In production, use Alembic migrations instead.
        """
        from .models import Base

        engine = await self.get_engine_async()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def drop_tables(self):
        """Drop all tables (for testing only)."""
        from .models import Base

        engine = await self.get_engine_async()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        logger.info("Database tables dropped")

    async def close(self):
        """Close session factory for the CURRENT event loop only.

        Note: The shared engine is NOT closed here. Use close_all() to close
        the shared engine when shutting down the application.
        """
        loop_id = self._get_loop_id()

        # Remove session factory for current loop (engine is shared, don't close it)
        if loop_id in self._session_factories:
            del self._session_factories[loop_id]

        # Remove backwards-compat engine reference
        if loop_id in self._engines:
            del self._engines[loop_id]

        logger.debug(f"Session factory closed for loop {loop_id}")

    async def close_shared_engine(self):
        """Close all per-loop engines and connectors. Call this on application shutdown."""
        self._shutdown = True

        # Close all per-loop engines
        for loop_id, engine in list(self._engines.items()):
            try:
                await engine.dispose()
                logger.debug(f"Disposed engine for loop {loop_id}")
            except Exception as e:
                logger.debug(f"Error disposing engine for loop {loop_id}: {e}")
        self._engines.clear()

        # Close all per-loop connectors
        for loop_id, connector in list(self._connectors.items()):
            if connector is not None:
                try:
                    if hasattr(connector, 'close_async'):
                        await connector.close_async()
                    else:
                        connector.close()
                    logger.debug(f"Closed connector for loop {loop_id}")
                except Exception as e:
                    logger.debug(f"Error closing connector for loop {loop_id}: {e}")
        self._connectors.clear()

        # Clear all session factories
        self._session_factories.clear()

        logger.info("All database engines closed")

    def close_sync(self, loop_id: Optional[int] = None):
        """
        Close resources for a specific event loop (sync version).

        Used by background threads to clean up their own resources
        before their event loop closes.

        Args:
            loop_id: The loop ID to close. If None, uses current loop.
        """
        if loop_id is None:
            loop_id = self._get_loop_id()

        # Close connector synchronously
        if loop_id in self._connectors and self._connectors[loop_id]:
            try:
                self._connectors[loop_id].close()
            except Exception as e:
                logger.debug(f"Error closing connector: {e}")
            del self._connectors[loop_id]

        # Dispose engine pool properly
        if loop_id in self._engines:
            engine = self._engines[loop_id]
            del self._engines[loop_id]

            # Try to dispose the underlying pool synchronously
            try:
                pool = engine.pool
                if pool:
                    pool.dispose()
                    logger.debug(f"Disposed connection pool synchronously for loop {loop_id}")
            except Exception as e:
                logger.debug(f"Could not dispose pool synchronously: {e}")

        if loop_id in self._session_factories:
            del self._session_factories[loop_id]

        logger.debug(f"Closed database resources for loop {loop_id}")

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics for monitoring.

        Returns:
            Dictionary with pool stats per event loop.
        """
        stats = {
            "pools_count": len(self._engines),
            "shutdown_mode": self._shutdown,
            "pools": {}
        }

        for loop_id, engine in self._engines.items():
            try:
                pool = engine.pool
                if pool:
                    stats["pools"][str(loop_id)] = {
                        "size": pool.size(),
                        "checked_out": pool.checkedout(),
                        "overflow": pool.overflow(),
                        "checked_in": pool.checkedin(),
                    }
            except Exception as e:
                stats["pools"][str(loop_id)] = {"error": str(e)}

        return stats

    async def close_all(self):
        """
        Close the shared engine and all session factories.

        Use this at application shutdown to ensure Cloud SQL connector
        is properly cleaned up.
        """
        # Close the shared engine (handles connector too)
        await self.close_shared_engine()

        # Clear backwards-compat connectors
        self._connectors.clear()

        self._initialized = False

        logger.info("All database connections closed")


# Global database manager instance
db = DatabaseManager()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection helper for FastAPI/similar frameworks.

    Usage:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    """
    async with db.session() as session:
        yield session
