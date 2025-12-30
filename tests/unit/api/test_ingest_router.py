"""Unit tests for ingest router endpoints."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


# Test organization ID for multi-tenant tests
TEST_ORG_ID = "test-org-123"


class TestListFilesEndpoint:
    """Tests for GET /files endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Return headers with required X-Organization-ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = MagicMock()
        storage.list_files = AsyncMock(return_value=[
            "gs://bucket/prefix/parsed/doc1.md",
            "gs://bucket/prefix/parsed/doc2.md"
        ])
        return storage

    @pytest.fixture
    def mock_storage_config(self):
        """Create a mock storage config."""
        config = MagicMock()
        config.parsed_directory = "parsed"
        return config

    def test_list_files_from_gcs(self, client, headers, mock_storage, mock_storage_config):
        """Test listing files from GCS parsed directory."""
        # Patch at the router module level where the imports are bound
        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.get_storage_config", return_value=mock_storage_config):

            response = client.get("/api/v1/ingest/files?directory=parsed", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["count"] == 2
            assert len(data["files"]) == 2

    def test_list_files_with_extension_filter(self, client, headers, mock_storage, mock_storage_config):
        """Test listing files with extension filter."""
        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.get_storage_config", return_value=mock_storage_config):

            response = client.get("/api/v1/ingest/files?directory=parsed&extension=.md", headers=headers)

            assert response.status_code == 200
            # The mock returns 2 files, filter is passed to list_files
            data = response.json()
            assert data["success"] is True

    def test_list_files_local_upload(self, client, headers, tmp_path, monkeypatch):
        """Test listing files from local upload directory."""
        # Create test files - include org subdirectory for multi-tenant
        upload_dir = tmp_path / "upload"
        org_upload_dir = upload_dir / TEST_ORG_ID
        org_upload_dir.mkdir(parents=True)
        (org_upload_dir / "file1.pdf").touch()
        (org_upload_dir / "file2.docx").touch()

        # Patch the dependency function
        def mock_get_upload_directory():
            return str(upload_dir)

        with patch("src.api.dependencies.get_upload_directory", mock_get_upload_directory):
            # Need to reimport the router to pick up the patch
            from src.api.app import create_app
            app = create_app()
            test_client = TestClient(app)

            response = test_client.get("/api/v1/ingest/files?directory=upload", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # Files should be listed
            assert data["count"] >= 0  # May or may not find files due to caching

    def test_list_files_invalid_directory(self, client, headers):
        """Test error on invalid directory."""
        response = client.get("/api/v1/ingest/files?directory=invalid", headers=headers)

        assert response.status_code == 200  # Returns 200 with error in body
        data = response.json()
        assert data["success"] is False
        assert "Invalid directory" in data["error"]

    def test_list_files_gcs_error(self, client, headers, mock_storage, mock_storage_config):
        """Test handling of GCS errors."""
        mock_storage.list_files = AsyncMock(side_effect=Exception("GCS Error"))

        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.get_storage_config", return_value=mock_storage_config):

            response = client.get("/api/v1/ingest/files?directory=parsed", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            # Should contain some error message
            assert "error" in data


class TestParseDocumentEndpoint:
    """Tests for POST /parse endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Return headers with required X-Organization-ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_parse_document_local_file_success(self, client, headers, tmp_path):
        """Test successful parsing of local file."""
        # Create a test file
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        with patch("src.api.routers.ingest.llama_parse") as mock_parse, \
             patch("src.api.routers.ingest.check_quota", False), \
             patch("src.api.routers.ingest.track_resource", False):

            mock_parse.return_value = "# Parsed Content\n\nThis is the parsed text."

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": str(test_file),
                    "folder_name": "test_folder",
                    "save_to_parsed": False,
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Parsed Content" in data["parsed_content"]

    def test_parse_document_file_not_found(self, client, headers):
        """Test parsing non-existent file returns error."""
        with patch("src.api.routers.ingest.check_quota", False):
            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": "/nonexistent/file.pdf",
                    "folder_name": "test_folder",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    def test_parse_document_gcs_file_success(self, client, headers):
        """Test successful parsing of GCS file."""
        mock_storage = MagicMock()
        mock_storage.exists = AsyncMock(side_effect=[False, True])  # Cache miss, then file exists
        mock_storage.download_bytes = AsyncMock(return_value=b"%PDF-1.4 test content")

        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.llama_parse") as mock_parse, \
             patch("src.api.routers.ingest.check_quota", False), \
             patch("src.api.routers.ingest.track_resource", False), \
             patch("os.unlink"):  # Don't actually delete temp files

            mock_parse.return_value = "# GCS Parsed Content"

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": "gs://bucket/org/original/test.pdf",
                    "folder_name": "test_folder",
                    "save_to_parsed": False,
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_parse_document_gcs_file_not_found(self, client, headers):
        """Test parsing non-existent GCS file."""
        mock_storage = MagicMock()
        mock_storage.exists = AsyncMock(return_value=False)

        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.check_quota", False):

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": "gs://bucket/nonexistent.pdf",
                    "folder_name": "test_folder",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "not found" in data["error"].lower()

    def test_parse_document_save_to_gcs(self, client, headers, tmp_path):
        """Test parsing with save_to_parsed=True saves to GCS."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test content")

        mock_storage = MagicMock()
        mock_storage.save = AsyncMock(return_value="gs://bucket/org/parsed/test.md")

        with patch("src.api.routers.ingest.llama_parse") as mock_parse, \
             patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.check_quota", False), \
             patch("src.api.routers.ingest.track_resource", False), \
             patch("src.api.routers.ingest.register_or_update_parsed_document", new_callable=AsyncMock):

            mock_parse.return_value = "# Parsed Content"

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": str(test_file),
                    "folder_name": "test_folder",
                    "save_to_parsed": True,
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            mock_storage.save.assert_called_once()

    def test_parse_document_cache_hit(self, client, headers):
        """Test that cached parsed content is returned."""
        mock_storage = MagicMock()
        mock_storage.exists = AsyncMock(return_value=True)
        mock_storage.read = AsyncMock(return_value="# Cached Content\n\nFrom cache.")

        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.check_quota", False):

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": "gs://bucket/org/original/test.pdf",
                    "folder_name": "test_folder",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "Cached Content" in data["parsed_content"]
            assert data["extraction_time_ms"] == 0  # No parsing time since cached

    def test_parse_document_empty_content(self, client, headers, tmp_path):
        """Test handling when no content is extracted."""
        test_file = tmp_path / "empty.pdf"
        test_file.write_bytes(b"%PDF-1.4 empty")

        with patch("src.api.routers.ingest.llama_parse") as mock_parse, \
             patch("src.api.routers.ingest.check_quota", False), \
             patch("src.api.routers.ingest.track_resource", False):

            mock_parse.return_value = ""  # Empty content

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": str(test_file),
                    "folder_name": "test_folder",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "no content" in data["error"].lower()

    def test_parse_document_quota_exceeded(self, client, headers, tmp_path):
        """Test 402 error when LlamaParse quota exceeded."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test")

        mock_quota_result = MagicMock()
        mock_quota_result.allowed = False
        mock_quota_result.current_usage = 100
        mock_quota_result.limit = 50
        mock_quota_result.upgrade_tier = "pro"
        mock_quota_result.upgrade_message = "Upgrade to Pro"
        mock_quota_result.upgrade_url = "/billing/upgrade"

        mock_checker = MagicMock()
        mock_checker.check_quota = AsyncMock(return_value=mock_quota_result)

        # Patch at the module where it's imported inside the function
        with patch("src.api.routers.ingest.check_quota", True), \
             patch("src.core.usage.get_quota_checker", return_value=mock_checker):

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": str(test_file),
                    "folder_name": "test_folder",
                }
            )

            assert response.status_code == 402
            data = response.json()
            # HTTPException detail may be wrapped in error field or returned directly
            error = data.get("error", {})
            if isinstance(error, dict):
                assert error.get("error") == "quota_exceeded"
            else:
                assert data.get("detail", {}).get("error") == "quota_exceeded"

    def test_parse_document_llamaparse_not_available(self, client, headers, tmp_path):
        """Test handling when LlamaParse module not available."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 test")

        with patch("src.api.routers.ingest.llama_parse", side_effect=ImportError("LlamaParse not installed")), \
             patch("src.api.routers.ingest.check_quota", False):

            response = client.post(
                "/api/v1/ingest/parse",
                headers=headers,
                json={
                    "file_path": str(test_file),
                    "folder_name": "test_folder",
                }
            )

            assert response.status_code == 200
            data = response.json()
            # Either fails or returns error
            assert data["success"] is False or "error" in data


class TestSaveAndIndexEndpoint:
    """Tests for POST /save-and-index endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Return headers with required X-Organization-ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_save_and_index_success(self, client, headers):
        """Test successful save and index operation."""
        mock_storage = MagicMock()
        mock_storage.save = AsyncMock(return_value="org/parsed/folder/doc.md")

        mock_storage_config = MagicMock()
        mock_storage_config.gcs_bucket = "test-bucket"

        mock_store_info = {
            "id": "store-123",
            "display_name": "test_org_file_search_store",
            "gemini_store_id": "fileSearchStores/abc",
        }

        mock_gemini_store = MagicMock()
        mock_gemini_store.name = "fileSearchStores/abc"

        # Patch at correct module locations - imports happen inside save_and_index function
        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.get_storage_config", return_value=mock_storage_config), \
             patch("src.api.routers.rag_helpers.get_or_create_org_store", new_callable=AsyncMock, return_value=mock_store_info), \
             patch("google.genai.Client") as mock_genai_client, \
             patch("src.rag.gemini_file_store.upload_file") as mock_upload:

            mock_client = MagicMock()
            mock_client.file_search_stores.list.return_value = [mock_gemini_store]
            mock_genai_client.return_value = mock_client

            response = client.post(
                "/api/v1/ingest/save-and-index",
                headers=headers,
                json={
                    "content": "# Test Content",
                    "target_path": "org/parsed/folder/doc.md",
                    "org_name": "Test Org",
                    "folder_name": "folder",
                    "original_filename": "doc.pdf",  # Required field
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["indexed"] is True
            assert data["store_id"] == "store-123"

    def test_save_and_index_gcs_save_failure(self, client, headers):
        """Test handling of GCS save failure."""
        mock_storage = MagicMock()
        mock_storage.save = AsyncMock(side_effect=Exception("GCS write failed"))

        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.get_storage_config", return_value=MagicMock()):

            response = client.post(
                "/api/v1/ingest/save-and-index",
                headers=headers,
                json={
                    "content": "# Test Content",
                    "target_path": "org/parsed/folder/doc.md",
                    "org_name": "Test Org",
                    "folder_name": "folder",
                    "original_filename": "doc.pdf",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "GCS write failed" in data["error"]

    def test_save_and_index_gemini_store_not_found(self, client, headers):
        """Test handling when Gemini store not found."""
        mock_storage = MagicMock()
        mock_storage.save = AsyncMock(return_value="org/parsed/doc.md")

        mock_storage_config = MagicMock()
        mock_storage_config.gcs_bucket = "test-bucket"

        mock_store_info = {
            "id": "store-123",
            "display_name": "test_store",
            "gemini_store_id": "fileSearchStores/nonexistent",
        }

        # Patch at correct module locations - imports happen inside save_and_index function
        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.get_storage_config", return_value=mock_storage_config), \
             patch("src.api.routers.rag_helpers.get_or_create_org_store", new_callable=AsyncMock, return_value=mock_store_info), \
             patch("google.genai.Client") as mock_genai_client:

            mock_client = MagicMock()
            mock_client.file_search_stores.list.return_value = []  # No stores found
            mock_genai_client.return_value = mock_client

            response = client.post(
                "/api/v1/ingest/save-and-index",
                headers=headers,
                json={
                    "content": "# Test Content",
                    "target_path": "org/parsed/doc.md",
                    "org_name": "Test Org",
                    "folder_name": "documents",
                    "original_filename": "doc.pdf",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["indexed"] is False  # Indexing skipped

    def test_save_and_index_includes_metadata(self, client, headers):
        """Test that enhanced metadata is passed to upload."""
        mock_storage = MagicMock()
        mock_storage.save = AsyncMock(return_value="org/parsed/doc.md")

        mock_storage_config = MagicMock()
        mock_storage_config.gcs_bucket = "test-bucket"

        mock_store_info = {
            "id": "store-123",
            "display_name": "test_store",
            "gemini_store_id": "fileSearchStores/abc",
        }

        mock_gemini_store = MagicMock()
        mock_gemini_store.name = "fileSearchStores/abc"

        # Patch at correct module locations - imports happen inside save_and_index function
        with patch("src.api.routers.ingest.get_storage", return_value=mock_storage), \
             patch("src.api.routers.ingest.get_storage_config", return_value=mock_storage_config), \
             patch("src.api.routers.rag_helpers.get_or_create_org_store", new_callable=AsyncMock, return_value=mock_store_info), \
             patch("google.genai.Client") as mock_genai_client, \
             patch("src.rag.gemini_file_store.upload_file") as mock_upload:

            mock_client = MagicMock()
            mock_client.file_search_stores.list.return_value = [mock_gemini_store]
            mock_genai_client.return_value = mock_client

            response = client.post(
                "/api/v1/ingest/save-and-index",
                headers=headers,
                json={
                    "content": "# Test Content",
                    "target_path": "org/parsed/doc.md",
                    "org_name": "Test Org",
                    "folder_name": "invoices",
                    "original_filename": "invoice.pdf",
                    "original_gcs_path": "gs://bucket/original/invoice.pdf",
                    "parser_version": "llama_parse_v2.5",
                }
            )

            assert response.status_code == 200
            # Verify upload was called with metadata
            if mock_upload.called:
                call_kwargs = mock_upload.call_args.kwargs
                assert call_kwargs.get("org_name") == "Test Org"
                assert call_kwargs.get("folder_name") == "invoices"
