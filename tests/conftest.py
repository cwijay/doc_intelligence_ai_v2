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


# =============================================================================
# Bulk Processing Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_bulk_singletons():
    """Reset bulk module singletons before and after each test."""
    # Reset before test
    try:
        from src.bulk.config import reset_bulk_config
        from src.bulk.folder_manager import reset_folder_manager
        reset_bulk_config()
        reset_folder_manager()
    except ImportError:
        pass

    yield

    # Reset after test
    try:
        from src.bulk.config import reset_bulk_config
        from src.bulk.folder_manager import reset_folder_manager
        reset_bulk_config()
        reset_folder_manager()
    except ImportError:
        pass


@pytest.fixture
def mock_bulk_config():
    """Create mock bulk processing config with test defaults."""
    from src.bulk.config import BulkProcessingConfig
    return BulkProcessingConfig(
        max_documents_per_folder=10,
        max_file_size_mb=50,
        concurrent_documents=3,
        parse_timeout_seconds=300,
        index_timeout_seconds=60,
        generation_timeout_seconds=300,
        job_timeout_seconds=3600,
        max_retries_per_document=3,
        auto_start_delay_seconds=60,
        auto_start_min_documents=1,
        webhook_enabled=True,
        webhook_secret=None,
    )


@pytest.fixture
def mock_folder_manager():
    """Create mock folder manager with AsyncMock methods."""
    from src.bulk.schemas import BulkFolderInfo, SignedUrlInfo
    from datetime import datetime

    manager = MagicMock()
    manager.create_folder = AsyncMock(return_value=BulkFolderInfo(
        folder_name="test-folder",
        gcs_path="gs://test-bucket/test-org/bulk/test-folder",
        document_count=0,
        total_size_bytes=0,
        created_at=datetime.utcnow(),
        org_id="test-org",
    ))
    manager.get_folder_info = AsyncMock(return_value=BulkFolderInfo(
        folder_name="test-folder",
        gcs_path="gs://test-bucket/test-org/bulk/test-folder",
        document_count=3,
        total_size_bytes=1024000,
        created_at=datetime.utcnow(),
        org_id="test-org",
    ))
    manager.list_folders = AsyncMock(return_value=[])
    manager.list_documents = AsyncMock(return_value=[])
    manager.generate_upload_urls = AsyncMock(return_value=[])
    manager.validate_folder_limit = AsyncMock(return_value=(True, "Folder is valid"))
    manager.folder_exists = AsyncMock(return_value=False)
    manager._is_supported_file = MagicMock(return_value=True)
    manager._get_content_type = MagicMock(return_value="application/pdf")
    return manager


@pytest.fixture
def mock_bulk_service():
    """Create mock bulk job service with AsyncMock methods."""
    from src.bulk.schemas import BulkJobInfo, BulkJobStatus, ProcessingOptions
    from datetime import datetime

    service = MagicMock()

    sample_job = BulkJobInfo(
        id="job-123",
        organization_id="test-org",
        folder_name="test-folder",
        source_path="gs://test-bucket/test-org/bulk/test-folder",
        total_documents=5,
        completed_count=0,
        failed_count=0,
        skipped_count=0,
        status=BulkJobStatus.PENDING,
        options=ProcessingOptions(),
        created_at=datetime.utcnow(),
    )

    service.create_job = AsyncMock(return_value=sample_job)
    service.start_job_processing = AsyncMock()
    service.process_single_document = AsyncMock()
    service.finalize_job = AsyncMock()
    service.cancel_job = AsyncMock(return_value=True)
    service.retry_failed_documents = AsyncMock(return_value=0)
    service.get_job_status = AsyncMock(return_value=sample_job)
    service.list_jobs = AsyncMock(return_value=[sample_job])
    return service


@pytest.fixture
def mock_bulk_queue():
    """Create mock bulk job queue."""
    queue = MagicMock()
    queue.enqueue = MagicMock()
    queue.start = MagicMock()
    queue.shutdown = MagicMock()
    queue.is_running = True
    return queue


@pytest.fixture
def sample_bulk_job_dict():
    """Sample bulk job dictionary for testing."""
    from datetime import datetime
    return {
        "id": "job-123",
        "organization_id": "test-org",
        "folder_name": "test-folder",
        "source_path": "gs://test-bucket/test-org/bulk/test-folder",
        "total_documents": 5,
        "completed_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "status": "pending",
        "options": {
            "generate_summary": True,
            "generate_faqs": True,
            "generate_questions": True,
            "num_faqs": 10,
            "num_questions": 10,
            "summary_max_words": 500,
        },
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def sample_document_item_dict():
    """Sample document item dictionary for testing."""
    from datetime import datetime
    return {
        "id": "doc-123",
        "bulk_job_id": "job-123",
        "original_path": "gs://test-bucket/test-org/bulk/test-folder/test.pdf",
        "original_filename": "test.pdf",
        "parsed_path": None,
        "status": "pending",
        "error_message": None,
        "retry_count": 0,
        "parse_time_ms": None,
        "index_time_ms": None,
        "generation_time_ms": None,
        "total_time_ms": None,
        "token_usage": 0,
        "llamaparse_pages": 0,
        "content_hash": None,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def mock_bulk_repository():
    """Mock bulk_repository module for testing."""
    mock_repo = MagicMock()

    # Job operations
    mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
    mock_repo.get_bulk_job = AsyncMock(return_value=None)
    mock_repo.list_bulk_jobs = AsyncMock(return_value=[])
    mock_repo.count_bulk_jobs = AsyncMock(return_value=0)
    mock_repo.update_bulk_job_status = AsyncMock(return_value=True)
    mock_repo.increment_job_completed = AsyncMock(return_value=True)
    mock_repo.increment_job_failed = AsyncMock(return_value=True)
    mock_repo.increment_job_skipped = AsyncMock(return_value=True)
    mock_repo.increment_total_documents = AsyncMock(return_value=True)
    mock_repo.find_active_job_for_folder = AsyncMock(return_value=None)

    # Document operations
    mock_repo.create_document_item = AsyncMock(return_value={"id": "doc-123"})
    mock_repo.get_document_item = AsyncMock(return_value=None)
    mock_repo.get_document_item_by_path = AsyncMock(return_value=None)
    mock_repo.get_all_document_items = AsyncMock(return_value=[])
    mock_repo.get_pending_documents = AsyncMock(return_value=[])
    mock_repo.get_failed_documents = AsyncMock(return_value=[])
    mock_repo.count_documents_in_job = AsyncMock(return_value=0)
    mock_repo.count_in_progress_documents = AsyncMock(return_value=0)
    mock_repo.update_document_item = AsyncMock(return_value=True)
    mock_repo.reset_document_for_retry = AsyncMock(return_value=True)
    mock_repo.get_latest_document_in_job = AsyncMock(return_value=None)

    return mock_repo


@pytest.fixture
def mock_gcs_for_bulk():
    """Mock GCS client for bulk folder operations."""
    with patch("google.cloud.storage.Client") as mock_client:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()

        # Configure blob behavior
        mock_blob.exists.return_value = True
        mock_blob.upload_from_string = MagicMock()
        mock_blob.generate_signed_url.return_value = "https://storage.googleapis.com/signed-url"
        mock_blob.size = 1024
        mock_blob.name = "test-org/bulk/test-folder/test.pdf"

        # Configure bucket behavior
        mock_bucket.blob.return_value = mock_blob
        mock_bucket.list_blobs.return_value = [mock_blob]

        mock_client.return_value.bucket.return_value = mock_bucket

        yield {
            "client": mock_client,
            "bucket": mock_bucket,
            "blob": mock_blob,
        }
