"""Shared test fixtures and configuration."""

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Async Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Environment Variable Fixtures
# =============================================================================

@pytest.fixture
def mock_gcs_env(monkeypatch):
    """Set up mock GCS environment variables."""
    monkeypatch.setenv("GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("GCS_PREFIX", "test-prefix")
    monkeypatch.setenv("PARSED_DIRECTORY", "parsed")
    monkeypatch.setenv("GENERATED_DIRECTORY", "generated")


@pytest.fixture
def clean_env(monkeypatch):
    """Clear GCS environment variables."""
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("GCS_PREFIX", raising=False)
    monkeypatch.delenv("PARSED_DIRECTORY", raising=False)
    monkeypatch.delenv("GENERATED_DIRECTORY", raising=False)


# =============================================================================
# Storage Singleton Reset
# =============================================================================

@pytest.fixture(autouse=True)
def reset_storage_singleton():
    """Reset storage singleton before and after each test."""
    # Reset before test
    try:
        from src.storage.config import reset_storage
        reset_storage()
    except ImportError:
        pass

    yield

    # Reset after test
    try:
        from src.storage.config import reset_storage
        reset_storage()
    except ImportError:
        pass


# =============================================================================
# GCS Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_blob():
    """Create a mock GCS blob."""
    blob = MagicMock()
    blob.name = "test-prefix/parsed/test.md"
    blob.exists.return_value = True
    blob.download_as_text.return_value = "# Test Content"
    blob.upload_from_string = MagicMock()
    blob.delete = MagicMock()
    return blob


@pytest.fixture
def mock_bucket(mock_blob):
    """Create a mock GCS bucket."""
    bucket = MagicMock()
    bucket.blob.return_value = mock_blob
    return bucket


@pytest.fixture
def mock_gcs_client(mock_bucket):
    """Create a mock GCS client."""
    client = MagicMock()
    client.bucket.return_value = mock_bucket
    client.list_blobs.return_value = []
    return client


@pytest.fixture
def mock_storage_client(mock_gcs_client):
    """Patch google.cloud.storage.Client and reset the singleton."""
    # Reset the module-level singleton before patching
    import src.storage.gcs as gcs_module
    gcs_module._gcs_client = None

    with patch("src.storage.gcs.storage.Client", return_value=mock_gcs_client):
        yield mock_gcs_client

    # Reset after test
    gcs_module._gcs_client = None


# =============================================================================
# GCSStorage Fixtures
# =============================================================================

@pytest.fixture
def gcs_storage(mock_storage_client, mock_bucket, mock_gcs_env):
    """Create a GCSStorage instance with mocked client."""
    from src.storage.gcs import GCSStorage
    storage = GCSStorage(bucket_name="test-bucket", prefix="test-prefix")
    # Ensure the storage instance uses our mock bucket
    storage._bucket = mock_bucket
    storage._client = mock_storage_client
    return storage


# =============================================================================
# Async Mock Helpers
# =============================================================================

@pytest.fixture
def async_mock():
    """Create an AsyncMock for async method testing."""
    return AsyncMock()


# =============================================================================
# Temporary Directory Fixtures
# =============================================================================

@pytest.fixture
def temp_upload_dir(tmp_path):
    """Create a temporary upload directory."""
    upload_dir = tmp_path / "upload"
    upload_dir.mkdir()
    return upload_dir


@pytest.fixture
def temp_parsed_dir(tmp_path):
    """Create a temporary parsed directory."""
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    return parsed_dir


# =============================================================================
# Sample Content Fixtures
# =============================================================================

@pytest.fixture
def sample_markdown_content():
    """Sample markdown content for tests."""
    return """# Test Document

## Overview
This is a test document for unit testing.

## Content
- Item 1
- Item 2
- Item 3
"""


@pytest.fixture
def sample_generated_content():
    """Sample generated content (summary, FAQs, questions)."""
    return {
        "document_name": "test.md",
        "generated_at": "2024-01-01T00:00:00Z",
        "summary": "This is a test summary.",
        "faqs": [
            {"question": "What is this?", "answer": "A test document."},
            {"question": "Why?", "answer": "For testing."}
        ],
        "questions": [
            {"question": "Describe the document.", "expected_answer": "A test doc.", "difficulty": "easy"}
        ],
        "model": "gemini-3-flash-preview"
    }


# =============================================================================
# Integration Test Markers
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires real GCS)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless RUN_INTEGRATION_TESTS is set."""
    if not os.getenv("RUN_INTEGRATION_TESTS"):
        skip_integration = pytest.mark.skip(
            reason="Set RUN_INTEGRATION_TESTS=1 to run integration tests"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


def pytest_sessionfinish(session, exitstatus):
    """
    Clean up all test resources after test session ends.

    Only runs cleanup for integration tests (when RUN_INTEGRATION_TESTS is set).
    Unit tests use mocks and don't create real resources.
    """
    if os.getenv("RUN_INTEGRATION_TESTS"):
        try:
            from tests.cleanup_utils import cleanup_all_test_resources
            cleanup_all_test_resources()
        except Exception as e:
            # Don't fail the test session on cleanup errors
            print(f"Warning: Test cleanup failed: {e}")


# =============================================================================
# Gemini Store Cleanup Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def gemini_test_stores():
    """
    Track Gemini stores created during tests for cleanup.

    Usage:
        def test_something(gemini_test_stores):
            store = create_file_search_store("test-my-store")
            gemini_test_stores.append(store)
            # ... test code ...
            # Store will be cleaned up after all tests in module
    """
    created_stores = []
    yield created_stores

    # Cleanup after all tests in module
    for store in created_stores:
        try:
            from src.rag.gemini_file_store import delete_store
            delete_store(store.name)
        except Exception:
            pass


def cleanup_gemini_stores_by_pattern(pattern: str):
    """
    Delete all Gemini stores matching a display_name pattern.

    Args:
        pattern: String pattern to match (e.g., "test_" for test stores)

    Usage:
        cleanup_gemini_stores_by_pattern("test_")
        cleanup_gemini_stores_by_pattern("_file_search_store")
    """
    try:
        from src.rag.gemini_file_store import list_all_stores, delete_store

        stores = list_all_stores()
        deleted = 0
        for store in stores:
            if pattern in store.display_name:
                try:
                    delete_store(store.name)
                    deleted += 1
                except Exception as e:
                    print(f"Failed to delete store {store.display_name}: {e}")
        return deleted
    except Exception as e:
        print(f"Cleanup error: {e}")
        return 0


def cleanup_db_stores_by_pattern(pattern: str):
    """
    Delete all database file_search_stores matching a display_name pattern.

    Args:
        pattern: String pattern to match

    Usage:
        cleanup_db_stores_by_pattern("test_")
    """
    import asyncio

    async def _cleanup():
        try:
            from src.db.connection import db
            from sqlalchemy import text

            async with db.session() as session:
                if session:
                    result = await session.execute(
                        text("DELETE FROM file_search_stores WHERE display_name LIKE :pattern"),
                        {"pattern": f"%{pattern}%"}
                    )
                    await session.commit()
                    return result.rowcount
        except Exception as e:
            print(f"DB cleanup error: {e}")
            return 0

    return asyncio.run(_cleanup())


# =============================================================================
# Database Cleanup Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def cleanup_db_after_module():
    """
    Clean up database test records after module tests complete.

    Usage:
        pytestmark = pytest.mark.usefixtures("cleanup_db_after_module")
    """
    yield

    # Cleanup after all tests in module
    try:
        import asyncio
        from src.db.repositories.audit_repository import delete_test_records

        loop = asyncio.get_event_loop()
        loop.run_until_complete(delete_test_records(prefix="test-"))
    except Exception:
        pass
