"""Unit tests for GCSStorage class."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from google.cloud.exceptions import NotFound, Forbidden
from google.api_core.exceptions import GoogleAPIError


class TestGCSStorageInit:
    """Tests for GCSStorage initialization."""

    def test_init_creates_client(self, mock_storage_client, mock_gcs_env):
        """Test that initialization creates a GCS client."""
        from src.storage.gcs import GCSStorage

        storage = GCSStorage(bucket_name="test-bucket", prefix="test-prefix")

        assert storage.bucket_name == "test-bucket"
        assert storage.prefix == "test-prefix"

    def test_init_strips_prefix_slashes(self, mock_storage_client, mock_gcs_env):
        """Test that prefix slashes are stripped."""
        from src.storage.gcs import GCSStorage

        storage = GCSStorage(bucket_name="test-bucket", prefix="/test-prefix/")

        assert storage.prefix == "test-prefix"


class TestGetBlobName:
    """Tests for _get_blob_name method."""

    def test_blob_name_with_prefix_and_directory(self, gcs_storage):
        """Test blob name construction with prefix and directory."""
        result = gcs_storage._get_blob_name("file.md", "subdir")
        assert result == "test-prefix/subdir/file.md"

    def test_blob_name_with_prefix_only(self, gcs_storage):
        """Test blob name with just prefix."""
        result = gcs_storage._get_blob_name("file.md", "")
        assert result == "test-prefix/file.md"

    def test_blob_name_without_prefix(self, mock_storage_client, mock_gcs_env):
        """Test blob name without prefix."""
        from src.storage.gcs import GCSStorage

        storage = GCSStorage(bucket_name="test-bucket", prefix="")
        result = storage._get_blob_name("file.md", "parsed")
        assert result == "parsed/file.md"


class TestSave:
    """Tests for save method."""

    @pytest.mark.asyncio
    async def test_save_uploads_content(self, gcs_storage, mock_bucket, mock_blob):
        """Test that save uploads content to GCS."""
        result = await gcs_storage.save("# Content", "test.md", "parsed")

        assert result == "gs://test-bucket/test-prefix/parsed/test.md"
        mock_blob.upload_from_string.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_sets_markdown_content_type(self, gcs_storage, mock_blob):
        """Test that .md files get text/markdown content type."""
        await gcs_storage.save("# Content", "test.md", "parsed")

        call_args = mock_blob.upload_from_string.call_args
        assert call_args[1]["content_type"] == "text/markdown"

    @pytest.mark.asyncio
    async def test_save_sets_json_content_type(self, gcs_storage, mock_blob):
        """Test that .json files get application/json content type."""
        await gcs_storage.save('{"key": "value"}', "test.json", "generated")

        call_args = mock_blob.upload_from_string.call_args
        assert call_args[1]["content_type"] == "application/json"

    @pytest.mark.asyncio
    async def test_save_raises_on_gcs_error(self, gcs_storage, mock_blob):
        """Test that GCS errors are raised."""
        mock_blob.upload_from_string.side_effect = GoogleAPIError("Upload failed")

        with pytest.raises(GoogleAPIError):
            await gcs_storage.save("content", "test.md", "parsed")


class TestRead:
    """Tests for read method."""

    @pytest.mark.asyncio
    async def test_read_returns_content(self, gcs_storage, mock_blob):
        """Test that read returns file content."""
        mock_blob.download_as_text.return_value = "# Test Content"

        result = await gcs_storage.read("parsed/test.md")

        assert result == "# Test Content"

    @pytest.mark.asyncio
    async def test_read_handles_gs_uri(self, gcs_storage, mock_bucket, mock_blob):
        """Test that read handles gs:// URIs."""
        mock_blob.download_as_text.return_value = "# Content"

        result = await gcs_storage.read("gs://test-bucket/test-prefix/parsed/test.md")

        assert result == "# Content"

    @pytest.mark.asyncio
    async def test_read_returns_none_on_not_found(self, gcs_storage, mock_blob):
        """Test that read returns None when file not found."""
        mock_blob.download_as_text.side_effect = NotFound("File not found")

        result = await gcs_storage.read("parsed/nonexistent.md")

        assert result is None

    @pytest.mark.asyncio
    async def test_read_raises_on_forbidden(self, gcs_storage, mock_blob):
        """Test that Forbidden errors are raised."""
        mock_blob.download_as_text.side_effect = Forbidden("Access denied")

        with pytest.raises(Forbidden):
            await gcs_storage.read("parsed/secret.md")


class TestExists:
    """Tests for exists method."""

    @pytest.mark.asyncio
    async def test_exists_returns_true(self, gcs_storage, mock_blob):
        """Test exists returns True for existing file."""
        mock_blob.exists.return_value = True

        result = await gcs_storage.exists("parsed/test.md")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false(self, gcs_storage, mock_blob):
        """Test exists returns False for non-existent file."""
        mock_blob.exists.return_value = False

        result = await gcs_storage.exists("parsed/nonexistent.md")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_handles_gs_uri(self, gcs_storage, mock_blob):
        """Test exists handles gs:// URIs."""
        mock_blob.exists.return_value = True

        result = await gcs_storage.exists("gs://test-bucket/test-prefix/parsed/test.md")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_error(self, gcs_storage, mock_blob):
        """Test exists returns False on GCS errors."""
        mock_blob.exists.side_effect = GoogleAPIError("Network error")

        result = await gcs_storage.exists("parsed/test.md")

        assert result is False


class TestListFiles:
    """Tests for list_files method."""

    @pytest.mark.asyncio
    async def test_list_files_returns_uris(self, gcs_storage, mock_gcs_client):
        """Test list_files returns list of URIs."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-prefix/parsed/file1.md"
        mock_blob2 = MagicMock()
        mock_blob2.name = "test-prefix/parsed/file2.md"
        mock_gcs_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        result = await gcs_storage.list_files("parsed")

        assert len(result) == 2
        assert "gs://test-bucket/test-prefix/parsed/file1.md" in result
        assert "gs://test-bucket/test-prefix/parsed/file2.md" in result

    @pytest.mark.asyncio
    async def test_list_files_filters_by_extension(self, gcs_storage, mock_gcs_client):
        """Test list_files filters by extension."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-prefix/parsed/file1.md"
        mock_blob2 = MagicMock()
        mock_blob2.name = "test-prefix/parsed/file2.txt"
        mock_gcs_client.list_blobs.return_value = [mock_blob1, mock_blob2]

        result = await gcs_storage.list_files("parsed", extension=".md")

        assert len(result) == 1
        assert "file1.md" in result[0]

    @pytest.mark.asyncio
    async def test_list_files_returns_empty_on_error(self, gcs_storage, mock_gcs_client):
        """Test list_files returns empty list on error."""
        mock_gcs_client.list_blobs.side_effect = GoogleAPIError("Error")

        result = await gcs_storage.list_files("parsed")

        assert result == []


class TestDelete:
    """Tests for delete method."""

    @pytest.mark.asyncio
    async def test_delete_returns_true(self, gcs_storage, mock_blob):
        """Test delete returns True on success."""
        result = await gcs_storage.delete("parsed/test.md")

        assert result is True
        mock_blob.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_not_found(self, gcs_storage, mock_blob):
        """Test delete returns False when file not found."""
        mock_blob.delete.side_effect = NotFound("Not found")

        result = await gcs_storage.delete("parsed/nonexistent.md")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_handles_gs_uri(self, gcs_storage, mock_blob):
        """Test delete handles gs:// URIs."""
        result = await gcs_storage.delete("gs://test-bucket/test-prefix/parsed/test.md")

        assert result is True


class TestGetUri:
    """Tests for get_uri method."""

    def test_get_uri_builds_from_relative_path(self, gcs_storage):
        """Test get_uri builds URI from relative path."""
        result = gcs_storage.get_uri("parsed/test.md")

        assert result == "gs://test-bucket/test-prefix/parsed/test.md"

    def test_get_uri_passes_through_gs_uri(self, gcs_storage):
        """Test get_uri passes through existing gs:// URIs."""
        uri = "gs://other-bucket/other-path/file.md"
        result = gcs_storage.get_uri(uri)

        assert result == uri
