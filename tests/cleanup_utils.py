"""
Utility functions for cleaning up test resources.

Provides centralized cleanup functions for:
- GCS files created during integration tests
- Gemini File Search stores
- Database test records
- Background executors

Usage in conftest.py:
    from tests.cleanup_utils import cleanup_all_test_resources

    def pytest_sessionfinish(session, exitstatus):
        cleanup_all_test_resources()
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def cleanup_gcs_test_files(prefix: str = "integration-tests"):
    """
    Delete all files with test prefix from GCS.

    Note: Individual test cleanup is handled in finally blocks.
    This is a safety net for any orphaned files.

    Args:
        prefix: Directory prefix to clean (default: "integration-tests")
    """
    try:
        from src.storage import get_storage

        storage = get_storage()

        # List all files in the integration-tests directory
        loop = asyncio.get_event_loop()
        files = loop.run_until_complete(storage.list_files(prefix))

        if files:
            logger.info(f"Cleaning up {len(files)} orphaned GCS test files...")
            for file_uri in files:
                try:
                    loop.run_until_complete(storage.delete(file_uri))
                except Exception as e:
                    logger.warning(f"Failed to delete {file_uri}: {e}")
            logger.info("GCS cleanup complete.")
        else:
            logger.debug("No orphaned GCS test files to clean up.")

    except Exception as e:
        logger.warning(f"GCS cleanup failed: {e}")


def cleanup_gemini_stores(display_name_prefix: str = "test-"):
    """
    Delete all Gemini File Search stores with test prefix.

    Args:
        display_name_prefix: Prefix to match store display names (default: "test-")
    """
    try:
        from src.rag.gemini_file_store import list_all_stores, delete_store

        stores = list_all_stores()
        test_stores = [
            s for s in stores
            if s.display_name and s.display_name.startswith(display_name_prefix)
        ]

        if test_stores:
            logger.info(f"Cleaning up {len(test_stores)} test Gemini stores...")
            for store in test_stores:
                try:
                    delete_store(store.name)
                    logger.debug(f"Deleted store: {store.display_name}")
                except Exception as e:
                    logger.warning(f"Failed to delete store {store.name}: {e}")
            logger.info("Gemini store cleanup complete.")
        else:
            logger.debug("No test Gemini stores to clean up.")

    except Exception as e:
        logger.warning(f"Gemini store cleanup failed: {e}")


def cleanup_database_records(prefix: str = "test-"):
    """
    Delete all database records with test prefix.

    Args:
        prefix: Prefix to match file_name/document_name (default: "test-")
    """
    try:
        from src.db.repositories.audit_repository import delete_test_records

        loop = asyncio.get_event_loop()
        counts = loop.run_until_complete(delete_test_records(prefix))

        total = sum(counts.values())
        if total > 0:
            logger.info(f"Cleaned up {total} database test records: {counts}")
        else:
            logger.debug("No database test records to clean up.")

    except Exception as e:
        logger.warning(f"Database cleanup failed: {e}")


def shutdown_executors():
    """
    Shutdown all background executors used during tests.

    This ensures all pending background tasks complete and
    resources are properly released.
    """
    try:
        from src.rag.gemini_file_store import shutdown_audit_executor

        shutdown_audit_executor(wait=True)
        logger.debug("Background executors shutdown complete.")

    except Exception as e:
        logger.warning(f"Executor shutdown failed: {e}")


def cleanup_all_test_resources(
    cleanup_gcs: bool = True,
    cleanup_gemini: bool = True,
    cleanup_db: bool = True,
    shutdown_exec: bool = True,
):
    """
    Clean up all test resources.

    This is the main entry point for test cleanup. Call this in
    pytest_sessionfinish hook.

    Args:
        cleanup_gcs: Whether to cleanup GCS files
        cleanup_gemini: Whether to cleanup Gemini stores
        cleanup_db: Whether to cleanup database records
        shutdown_exec: Whether to shutdown executors
    """
    logger.info("Starting test resource cleanup...")

    if cleanup_gcs:
        cleanup_gcs_test_files()

    if cleanup_gemini:
        cleanup_gemini_stores()

    if cleanup_db:
        cleanup_database_records()

    if shutdown_exec:
        shutdown_executors()

    logger.info("Test resource cleanup complete.")
