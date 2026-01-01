"""Unit tests for FolderManager."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path

from src.bulk.folder_manager import (
    FolderManager,
    get_folder_manager,
    reset_folder_manager,
)
from src.bulk.schemas import BulkFolderInfo, SignedUrlInfo
from src.bulk.config import BulkProcessingConfig


@pytest.fixture
def mock_storage():
    """Create mock storage instance."""
    storage = MagicMock()
    storage.bucket_name = "test-bucket"
    return storage


@pytest.fixture
def mock_gcs_bucket():
    """Create mock GCS bucket."""
    bucket = MagicMock()
    return bucket


@pytest.fixture
def mock_gcs_client(mock_gcs_bucket):
    """Create mock GCS client."""
    client = MagicMock()
    client.bucket.return_value = mock_gcs_bucket
    return client


class TestFolderManagerInit:
    """Tests for FolderManager initialization."""

    def test_init_with_config(self, mock_storage, mock_gcs_client):
        """Test initialization with custom config."""
        config = BulkProcessingConfig(max_documents_per_folder=50)

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager(config=config)

            assert manager.config.max_documents_per_folder == 50
            assert manager._bucket_name == "test-bucket"

    def test_init_default_config(self, mock_storage, mock_gcs_client):
        """Test initialization with default config."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()

            assert manager.config is not None
            assert manager._bucket is not None


class TestFolderManagerBulkPath:
    """Tests for _get_bulk_path helper method."""

    def test_get_bulk_path_format(self, mock_storage, mock_gcs_client):
        """Test bulk path format."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            path = manager._get_bulk_path("test-org", "invoices")

            assert path == "test-org/bulk/invoices"

    def test_get_full_gcs_uri(self, mock_storage, mock_gcs_client):
        """Test full GCS URI generation."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            uri = manager._get_full_gcs_uri("test-org/bulk/invoices")

            assert uri == "gs://test-bucket/test-org/bulk/invoices"


class TestFolderManagerCreateFolder:
    """Tests for FolderManager.create_folder()."""

    @pytest.mark.asyncio
    async def test_create_folder_success(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test successful folder creation."""
        mock_blob = MagicMock()
        mock_gcs_bucket.blob.return_value = mock_blob

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            # Mock the run_in_executor to run synchronously
            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                result = await manager.create_folder("test-org", "invoices")

            assert isinstance(result, BulkFolderInfo)
            assert result.folder_name == "invoices"
            assert result.org_id == "test-org"
            assert result.document_count == 0
            assert "test-bucket" in result.gcs_path

    @pytest.mark.asyncio
    async def test_create_folder_path_format(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test that folder path follows org/bulk/name format."""
        mock_blob = MagicMock()
        mock_gcs_bucket.blob.return_value = mock_blob

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)

                await manager.create_folder("my-org", "my-folder")

            # Check that blob was created with correct path
            mock_gcs_bucket.blob.assert_called_once()
            blob_path = mock_gcs_bucket.blob.call_args[0][0]
            assert blob_path == "my-org/bulk/my-folder/.folder_created"


class TestFolderManagerGetFolderInfo:
    """Tests for FolderManager.get_folder_info()."""

    @pytest.mark.asyncio
    async def test_get_folder_info_exists(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test getting info for existing folder."""
        # Create mock blobs
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-org/bulk/test-folder/.folder_created"
        mock_blob1.size = 0

        mock_blob2 = MagicMock()
        mock_blob2.name = "test-org/bulk/test-folder/doc.pdf"
        mock_blob2.size = 1024

        mock_gcs_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value=[mock_blob1, mock_blob2]
                )

                result = await manager.get_folder_info("test-org", "test-folder")

            assert result is not None
            assert result.folder_name == "test-folder"
            assert result.org_id == "test-org"

    @pytest.mark.asyncio
    async def test_get_folder_info_not_found(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test None return when folder not found."""
        mock_gcs_bucket.list_blobs.return_value = []

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=[])

                result = await manager.get_folder_info("test-org", "nonexistent")

            assert result is None


class TestFolderManagerListFolders:
    """Tests for FolderManager.list_folders()."""

    @pytest.mark.asyncio
    async def test_list_folders_success(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test listing folders for org."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                # Return empty prefixes for list_folders
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=[])

                result = await manager.list_folders("test-org")

            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_folders_empty(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test empty folder list."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=[])

                result = await manager.list_folders("test-org")

            assert result == []


class TestFolderManagerListDocuments:
    """Tests for FolderManager.list_documents()."""

    @pytest.mark.asyncio
    async def test_list_documents_success(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test listing documents in folder."""
        mock_blob = MagicMock()
        mock_blob.name = "test-org/bulk/test-folder/doc.pdf"

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=[mock_blob])

                result = await manager.list_documents("test-org", "test-folder")

            assert len(result) == 1
            assert "doc.pdf" in result[0]

    @pytest.mark.asyncio
    async def test_list_documents_filters_unsupported(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test filtering of unsupported file types."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-org/bulk/test-folder/doc.pdf"

        mock_blob2 = MagicMock()
        mock_blob2.name = "test-org/bulk/test-folder/script.exe"

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value=[mock_blob1, mock_blob2]
                )

                result = await manager.list_documents("test-org", "test-folder")

            # Only PDF should be in results
            assert len(result) == 1
            assert "doc.pdf" in result[0]

    @pytest.mark.asyncio
    async def test_list_documents_skips_hidden_files(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test that hidden files are skipped."""
        mock_blob1 = MagicMock()
        mock_blob1.name = "test-org/bulk/test-folder/doc.pdf"

        mock_blob2 = MagicMock()
        mock_blob2.name = "test-org/bulk/test-folder/.folder_created"

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value=[mock_blob1, mock_blob2]
                )

                result = await manager.list_documents("test-org", "test-folder")

            # Only visible doc.pdf should be in results
            assert len(result) == 1
            assert "doc.pdf" in result[0]


class TestFolderManagerGenerateUploadUrls:
    """Tests for FolderManager.generate_upload_urls()."""

    @pytest.mark.asyncio
    async def test_generate_urls_success(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test signed URL generation."""
        mock_blob = MagicMock()
        mock_blob.generate_signed_url.return_value = "https://signed-url.example.com"
        mock_gcs_bucket.blob.return_value = mock_blob

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value="https://signed-url.example.com"
                )

                result = await manager.generate_upload_urls(
                    org_id="test-org",
                    folder_name="test-folder",
                    filenames=["doc.pdf"],
                )

            assert len(result) == 1
            assert isinstance(result[0], SignedUrlInfo)
            assert result[0].filename == "doc.pdf"
            assert "signed-url" in result[0].signed_url

    @pytest.mark.asyncio
    async def test_generate_urls_filters_unsupported(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test that unsupported extensions are filtered."""
        mock_blob = MagicMock()
        mock_blob.generate_signed_url.return_value = "https://signed-url.example.com"
        mock_gcs_bucket.blob.return_value = mock_blob

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    return_value="https://signed-url.example.com"
                )

                result = await manager.generate_upload_urls(
                    org_id="test-org",
                    folder_name="test-folder",
                    filenames=["doc.pdf", "script.exe"],  # exe should be filtered
                )

            # Only PDF should have URL generated
            assert len(result) == 1
            assert result[0].filename == "doc.pdf"


class TestFolderManagerValidation:
    """Tests for FolderManager.validate_folder_limit()."""

    @pytest.mark.asyncio
    async def test_validate_folder_success(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test valid folder passes validation."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            # Mock get_folder_info to return valid folder
            manager.get_folder_info = AsyncMock(return_value=BulkFolderInfo(
                folder_name="test-folder",
                gcs_path="gs://test-bucket/test-org/bulk/test-folder",
                document_count=5,
                total_size_bytes=1024,
                org_id="test-org",
            ))

            is_valid, message = await manager.validate_folder_limit("test-org", "test-folder")

            assert is_valid is True
            assert "valid" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_folder_not_found(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test validation fails for missing folder."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            # Mock get_folder_info to return None
            manager.get_folder_info = AsyncMock(return_value=None)

            is_valid, message = await manager.validate_folder_limit("test-org", "nonexistent")

            assert is_valid is False
            assert "not found" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_folder_empty(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test validation fails for empty folder."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            # Mock get_folder_info to return empty folder
            manager.get_folder_info = AsyncMock(return_value=BulkFolderInfo(
                folder_name="test-folder",
                gcs_path="gs://test-bucket/test-org/bulk/test-folder",
                document_count=0,
                total_size_bytes=0,
                org_id="test-org",
            ))

            is_valid, message = await manager.validate_folder_limit("test-org", "test-folder")

            assert is_valid is False
            assert "no supported documents" in message.lower()

    @pytest.mark.asyncio
    async def test_validate_folder_exceeds_limit(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test validation fails when exceeding document limit."""
        config = BulkProcessingConfig(max_documents_per_folder=5)

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager(config=config)

            # Mock get_folder_info to return folder over limit
            manager.get_folder_info = AsyncMock(return_value=BulkFolderInfo(
                folder_name="test-folder",
                gcs_path="gs://test-bucket/test-org/bulk/test-folder",
                document_count=10,  # Over limit of 5
                total_size_bytes=1024,
                org_id="test-org",
            ))

            is_valid, message = await manager.validate_folder_limit("test-org", "test-folder")

            assert is_valid is False
            assert "10 documents" in message


class TestFolderManagerFolderExists:
    """Tests for FolderManager.folder_exists()."""

    @pytest.mark.asyncio
    async def test_folder_exists_true(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test True when folder exists."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_gcs_bucket.blob.return_value = mock_blob

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)

                result = await manager.folder_exists("test-org", "test-folder")

            assert result is True

    @pytest.mark.asyncio
    async def test_folder_exists_false(self, mock_storage, mock_gcs_client, mock_gcs_bucket):
        """Test False when folder doesn't exist."""
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_gcs_bucket.blob.return_value = mock_blob

        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client), \
             patch("src.bulk.folder_manager.get_executors") as mock_exec:

            mock_exec.return_value.io_executor = None

            manager = FolderManager()

            with patch("asyncio.get_running_loop") as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=False)

                result = await manager.folder_exists("test-org", "nonexistent")

            assert result is False


class TestFolderManagerHelpers:
    """Tests for helper methods."""

    def test_is_supported_file_pdf(self, mock_storage, mock_gcs_client):
        """Test PDF is supported."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._is_supported_file("document.pdf") is True

    def test_is_supported_file_docx(self, mock_storage, mock_gcs_client):
        """Test DOCX is supported."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._is_supported_file("document.docx") is True

    def test_is_supported_file_unsupported(self, mock_storage, mock_gcs_client):
        """Test unsupported extension returns False."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._is_supported_file("script.exe") is False

    def test_is_supported_file_hidden(self, mock_storage, mock_gcs_client):
        """Test hidden files are not supported."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._is_supported_file(".hidden.pdf") is False

    def test_get_content_type_pdf(self, mock_storage, mock_gcs_client):
        """Test content type for PDF."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._get_content_type("doc.pdf") == "application/pdf"

    def test_get_content_type_docx(self, mock_storage, mock_gcs_client):
        """Test content type for DOCX."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            content_type = manager._get_content_type("doc.docx")
            assert "wordprocessingml" in content_type

    def test_get_content_type_unknown(self, mock_storage, mock_gcs_client):
        """Test content type for unknown extension."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._get_content_type("file.xyz") == "application/octet-stream"

    def test_get_content_type_png(self, mock_storage, mock_gcs_client):
        """Test content type for PNG."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._get_content_type("image.png") == "image/png"

    def test_get_content_type_jpg(self, mock_storage, mock_gcs_client):
        """Test content type for JPG."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = FolderManager()
            assert manager._get_content_type("photo.jpg") == "image/jpeg"


class TestFolderManagerSingleton:
    """Tests for folder manager singleton."""

    def test_get_folder_manager_returns_instance(self, mock_storage, mock_gcs_client):
        """Test get_folder_manager returns an instance."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager = get_folder_manager()
            assert isinstance(manager, FolderManager)

    def test_get_folder_manager_same_instance(self, mock_storage, mock_gcs_client):
        """Test get_folder_manager returns same instance."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager1 = get_folder_manager()
            manager2 = get_folder_manager()
            assert manager1 is manager2

    def test_reset_folder_manager(self, mock_storage, mock_gcs_client):
        """Test reset clears singleton."""
        with patch("src.bulk.folder_manager.get_storage", return_value=mock_storage), \
             patch("src.bulk.folder_manager.storage.Client", return_value=mock_gcs_client):

            manager1 = get_folder_manager()
            reset_folder_manager()
            manager2 = get_folder_manager()
            assert manager1 is not manager2
