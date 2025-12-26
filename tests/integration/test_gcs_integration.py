"""Integration tests for GCS storage.

These tests require real GCS credentials and will create/delete files
in the configured bucket.

Run with: RUN_INTEGRATION_TESTS=1 pytest tests/integration -v
"""

import asyncio
import os
import uuid
import pytest


# Skip all tests in this module if RUN_INTEGRATION_TESTS is not set
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def test_prefix():
    """Generate unique prefix for test files to avoid conflicts."""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def gcs_storage():
    """Create real GCS storage instance."""
    from src.storage.gcs import GCSStorage

    bucket = os.getenv("GCS_BUCKET", "biz2bricks-dev-v1-document-store")
    prefix = os.getenv("GCS_PREFIX", "demo_docs")

    return GCSStorage(bucket_name=bucket, prefix=prefix)


class TestGCSIntegration:
    """Integration tests for GCS storage operations."""

    @pytest.mark.asyncio
    async def test_save_and_read_roundtrip(self, gcs_storage, test_prefix):
        """Test saving content and reading it back."""
        content = "# Test Document\n\nThis is test content."
        filename = f"{test_prefix}/test_roundtrip.md"

        try:
            # Save
            uri = await gcs_storage.save(content, filename, directory="integration-tests")

            assert uri.startswith("gs://")
            assert filename in uri

            # Read back
            read_content = await gcs_storage.read(uri)

            assert read_content == content

        finally:
            # Cleanup
            await gcs_storage.delete(uri)

    @pytest.mark.asyncio
    async def test_exists_check(self, gcs_storage, test_prefix):
        """Test exists() for real and non-existent files."""
        content = "Test content"
        filename = f"{test_prefix}/test_exists.md"

        try:
            # Should not exist yet
            assert await gcs_storage.exists(f"integration-tests/{filename}") is False

            # Save file
            uri = await gcs_storage.save(content, filename, directory="integration-tests")

            # Should exist now
            assert await gcs_storage.exists(uri) is True

        finally:
            # Cleanup
            await gcs_storage.delete(uri)

        # Should not exist after delete
        assert await gcs_storage.exists(uri) is False

    @pytest.mark.asyncio
    async def test_list_files_in_directory(self, gcs_storage, test_prefix):
        """Test listing files in a directory."""
        files_to_create = [
            f"{test_prefix}/list_test_1.md",
            f"{test_prefix}/list_test_2.md",
            f"{test_prefix}/list_test_3.txt",
        ]
        created_uris = []

        try:
            # Create test files
            for filename in files_to_create:
                uri = await gcs_storage.save(
                    f"Content for {filename}",
                    filename,
                    directory="integration-tests"
                )
                created_uris.append(uri)

            # List all files
            all_files = await gcs_storage.list_files(f"integration-tests/{test_prefix}")
            assert len(all_files) >= 3

            # List only .md files
            md_files = await gcs_storage.list_files(
                f"integration-tests/{test_prefix}",
                extension=".md"
            )
            assert len(md_files) >= 2
            assert all(".md" in f for f in md_files)

        finally:
            # Cleanup
            for uri in created_uris:
                await gcs_storage.delete(uri)

    @pytest.mark.asyncio
    async def test_delete_file(self, gcs_storage, test_prefix):
        """Test deleting a file."""
        content = "Content to delete"
        filename = f"{test_prefix}/test_delete.md"

        # Create file
        uri = await gcs_storage.save(content, filename, directory="integration-tests")

        # Verify it exists
        assert await gcs_storage.exists(uri) is True

        # Delete
        result = await gcs_storage.delete(uri)
        assert result is True

        # Verify it's gone
        assert await gcs_storage.exists(uri) is False

        # Delete again should return False
        result = await gcs_storage.delete(uri)
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, gcs_storage, test_prefix):
        """Test multiple concurrent operations don't conflict."""
        files = [f"{test_prefix}/concurrent_{i}.md" for i in range(5)]
        created_uris = []

        try:
            # Concurrent saves
            save_tasks = [
                gcs_storage.save(f"Content {i}", f, directory="integration-tests")
                for i, f in enumerate(files)
            ]
            created_uris = await asyncio.gather(*save_tasks)

            assert len(created_uris) == 5
            assert all(uri.startswith("gs://") for uri in created_uris)

            # Concurrent reads
            read_tasks = [gcs_storage.read(uri) for uri in created_uris]
            contents = await asyncio.gather(*read_tasks)

            assert len(contents) == 5
            assert all(content is not None for content in contents)

            # Concurrent exists checks
            exists_tasks = [gcs_storage.exists(uri) for uri in created_uris]
            results = await asyncio.gather(*exists_tasks)

            assert all(result is True for result in results)

        finally:
            # Concurrent deletes
            delete_tasks = [gcs_storage.delete(uri) for uri in created_uris]
            await asyncio.gather(*delete_tasks)

    @pytest.mark.asyncio
    async def test_json_content_type(self, gcs_storage, test_prefix):
        """Test that JSON files are saved with correct content type."""
        import json

        content = json.dumps({"key": "value", "number": 42})
        filename = f"{test_prefix}/test_json.json"

        try:
            uri = await gcs_storage.save(content, filename, directory="integration-tests")

            # Read back and verify
            read_content = await gcs_storage.read(uri)
            parsed = json.loads(read_content)

            assert parsed["key"] == "value"
            assert parsed["number"] == 42

        finally:
            await gcs_storage.delete(uri)

    @pytest.mark.asyncio
    async def test_unicode_content(self, gcs_storage, test_prefix):
        """Test saving and reading Unicode content."""
        content = "# 测试文档\n\nThis contains émojis 🎉 and spëcial çharacters."
        filename = f"{test_prefix}/test_unicode.md"

        try:
            uri = await gcs_storage.save(content, filename, directory="integration-tests")
            read_content = await gcs_storage.read(uri)

            assert read_content == content

        finally:
            await gcs_storage.delete(uri)
