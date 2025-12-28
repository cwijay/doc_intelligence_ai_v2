"""Unit tests for RAG router endpoints.

All tests use mocks to prevent real API calls and database operations.
No cleanup required as no real resources are created.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


# Test organization ID for multi-tenant tests
TEST_ORG_ID = "test-org-123"


@pytest.fixture(autouse=True)
def mock_gemini_globally():
    """
    Mock Gemini client globally to prevent real API calls.
    Applied to all tests in this module.
    """
    with patch("google.genai.Client") as mock_client:
        mock_instance = MagicMock()
        mock_instance.file_search_stores.list.return_value = []
        mock_client.return_value = mock_instance
        yield mock_client


class TestUploadToStoreEndpoint:
    """Tests for POST /stores/{store_id}/upload endpoint."""

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
    def mock_store_info(self):
        """Create mock store info dict."""
        return {
            "id": "store-123",
            "organization_id": TEST_ORG_ID,
            "gemini_store_id": "fileSearchStores/gemini-store-123",
            "display_name": "test_org_file_search_store",
        }

    @pytest.fixture
    def mock_gemini_store(self):
        """Create a mock Gemini store object."""
        store = MagicMock()
        store.name = "fileSearchStores/gemini-store-123"
        store.display_name = "test_org_file_search_store"
        return store

    def test_upload_with_auto_store_id_requires_org_name(self, client, headers):
        """Test that store_id='auto' requires org_name in request."""
        response = client.post(
            "/api/v1/rag/stores/auto/upload",
            headers=headers,
            json={
                "file_paths": ["/path/to/file.md"],
                # Missing org_name
            }
        )

        assert response.status_code == 400
        # Check the error is in the response body (could be in 'detail' or 'error')
        data = response.json()
        assert "org_name" in str(data).lower()

    def test_upload_with_auto_store_id_creates_store(
        self, client, headers, mock_store_info, mock_gemini_store, tmp_path
    ):
        """Test that store_id='auto' creates/gets org-specific store."""
        # Create actual test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test content")

        with patch("src.api.routers.rag.get_or_create_org_store", new_callable=AsyncMock) as mock_get_store, \
             patch("src.api.routers.rag.rag_repository.update_store_stats", new_callable=AsyncMock), \
             patch("src.rag.gemini_file_store.upload_file") as mock_upload, \
             patch("google.genai.Client") as mock_genai_client:

            mock_get_store.return_value = mock_store_info

            # Mock Gemini client
            mock_client_instance = MagicMock()
            mock_client_instance.file_search_stores.list.return_value = [mock_gemini_store]
            mock_genai_client.return_value = mock_client_instance

            response = client.post(
                "/api/v1/rag/stores/auto/upload",
                headers=headers,
                json={
                    "file_paths": [str(test_file)],
                    "org_name": "Test Org",
                }
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["store_id"] == "store-123"
            mock_get_store.assert_called_once_with(TEST_ORG_ID, "Test Org")

    def test_upload_with_explicit_store_id(
        self, client, headers, mock_store_info, mock_gemini_store, tmp_path
    ):
        """Test upload with explicit store_id validates ownership."""
        # Create actual test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test content")

        with patch("src.api.routers.rag.validate_store_ownership", new_callable=AsyncMock) as mock_validate, \
             patch("src.api.routers.rag.rag_repository.update_store_stats", new_callable=AsyncMock), \
             patch("src.rag.gemini_file_store.upload_file") as mock_upload, \
             patch("google.genai.Client") as mock_genai_client:

            mock_validate.return_value = mock_store_info

            # Mock Gemini client
            mock_client_instance = MagicMock()
            mock_client_instance.file_search_stores.list.return_value = [mock_gemini_store]
            mock_genai_client.return_value = mock_client_instance

            response = client.post(
                "/api/v1/rag/stores/store-123/upload",
                headers=headers,
                json={
                    "file_paths": [str(test_file)],
                }
            )

            assert response.status_code == 200
            mock_validate.assert_called_once_with("store-123", TEST_ORG_ID)

    def test_upload_includes_enhanced_metadata(
        self, client, headers, mock_store_info, mock_gemini_store, tmp_path
    ):
        """Test that upload passes enhanced metadata to upload_file."""
        # Create actual test file
        test_file = tmp_path / "doc.md"
        test_file.write_text("# Test content")

        with patch("src.api.routers.rag.get_or_create_org_store", new_callable=AsyncMock) as mock_get_store, \
             patch("src.api.routers.rag.rag_repository.update_store_stats", new_callable=AsyncMock), \
             patch("src.rag.gemini_file_store.upload_file") as mock_upload, \
             patch("google.genai.Client") as mock_genai_client:

            mock_get_store.return_value = mock_store_info

            # Mock Gemini client
            mock_client_instance = MagicMock()
            mock_client_instance.file_search_stores.list.return_value = [mock_gemini_store]
            mock_genai_client.return_value = mock_client_instance

            response = client.post(
                "/api/v1/rag/stores/auto/upload",
                headers=headers,
                json={
                    "file_paths": [str(test_file)],
                    "org_name": "Test Org",
                    "folder_name": "Test Folder",
                    "original_gcs_paths": ["gs://bucket/original/doc.pdf"],
                    "parser_version": "llama_parse_v2.5",
                }
            )

            assert response.status_code == 200

            # Verify upload_file was called with enhanced metadata
            mock_upload.assert_called_once()
            call_kwargs = mock_upload.call_args.kwargs

            assert call_kwargs["org_name"] == "Test Org"
            assert call_kwargs["content_hash"] is not None  # Hash should be computed
            assert call_kwargs["original_gcs_path"] == "gs://bucket/original/doc.pdf"
            assert call_kwargs["original_file_extension"] == ".pdf"
            assert call_kwargs["parser_version"] == "llama_parse_v2.5"


class TestContentHashComputation:
    """Tests for content hash computation helpers."""

    def test_compute_content_hash_local_file(self, tmp_path):
        """Test computing hash of local file."""
        from src.api.routers.rag_helpers import compute_content_hash

        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash_result = compute_content_hash(str(test_file))

        # Should be a valid SHA-256 hex string (64 chars)
        assert len(hash_result) == 64
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_compute_content_hash_consistent(self, tmp_path):
        """Test that same content produces same hash."""
        from src.api.routers.rag_helpers import compute_content_hash

        # Create two files with same content
        test_file1 = tmp_path / "test1.txt"
        test_file2 = tmp_path / "test2.txt"
        test_file1.write_text("Same content")
        test_file2.write_text("Same content")

        hash1 = compute_content_hash(str(test_file1))
        hash2 = compute_content_hash(str(test_file2))

        assert hash1 == hash2

    @pytest.mark.asyncio
    async def test_compute_gcs_content_hash(self):
        """Test computing hash of GCS file."""
        from src.api.routers.rag_helpers import compute_gcs_content_hash

        mock_storage = MagicMock()
        mock_storage.read = AsyncMock(return_value="Test content")

        with patch("src.storage.get_storage", return_value=mock_storage):
            hash_result = await compute_gcs_content_hash("gs://bucket/path/file.md")

            assert hash_result is not None
            assert len(hash_result) == 64


class TestGetOrCreateOrgStore:
    """Tests for get_or_create_org_store helper."""

    @pytest.mark.asyncio
    async def test_returns_existing_store_from_db(self):
        """Test that existing store from DB is returned."""
        from src.api.routers.rag_helpers import get_or_create_org_store

        mock_store_info = {
            "id": "store-123",
            "organization_id": "org-123",
            "gemini_store_id": "fileSearchStores/gemini-123",
            "display_name": "test_org_file_search_store",
        }

        with patch("src.api.routers.rag_helpers.rag_repository.get_store_by_display_name", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_store_info

            result = await get_or_create_org_store("org-123", "Test Org")

            assert result == mock_store_info
            mock_get.assert_called_once_with("test_org_file_search_store", organization_id="org-123")

    @pytest.mark.asyncio
    async def test_creates_new_store_when_not_in_db(self):
        """Test that new store is created when not in DB."""
        from src.api.routers.rag_helpers import get_or_create_org_store

        mock_gemini_store = MagicMock()
        mock_gemini_store.name = "fileSearchStores/new-gemini-store"

        mock_new_store_info = {
            "id": "new-store-123",
            "organization_id": "org-123",
            "gemini_store_id": "fileSearchStores/new-gemini-store",
            "display_name": "test_org_file_search_store",
        }

        with patch("src.api.routers.rag_helpers.rag_repository.get_store_by_display_name", new_callable=AsyncMock) as mock_get, \
             patch("src.rag.gemini_file_store.get_or_create_store_by_org_name", return_value=(mock_gemini_store, True)), \
             patch("src.api.routers.rag_helpers.rag_repository.create_store", new_callable=AsyncMock) as mock_create:

            mock_get.return_value = None  # Not in DB
            mock_create.return_value = mock_new_store_info

            result = await get_or_create_org_store("org-123", "Test Org")

            assert result == mock_new_store_info
            mock_create.assert_called_once()


class TestCreateStoreEndpoint:
    """Tests for POST /stores endpoint."""

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

    def test_create_store_new(self, client, headers):
        """Test creating a new store when org doesn't have one."""
        mock_gemini_store = MagicMock()
        mock_gemini_store.name = "fileSearchStores/new-store"

        new_store_data = {
            "id": "store-new",
            "display_name": "test_org_file_search_store",
            "gemini_store_id": "fileSearchStores/new-store",
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch("src.api.routers.rag.rag_repository.get_store_by_org", new_callable=AsyncMock, return_value=None), \
             patch("src.rag.gemini_file_store.create_file_search_store", return_value=mock_gemini_store), \
             patch("src.api.routers.rag.rag_repository.create_store", new_callable=AsyncMock, return_value=new_store_data):

            response = client.post(
                "/api/v1/rag/stores",
                headers=headers,
                json={"display_name": "Test Store"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["store_id"] == "store-new"

    def test_create_store_already_exists(self, client, headers):
        """Test that creating a store returns existing store if org already has one."""
        existing_store = {
            "id": "store-existing",
            "display_name": "test_org_file_search_store",
            "gemini_store_id": "fileSearchStores/existing",
            "created_at": "2024-01-01T00:00:00Z",
        }

        with patch("src.api.routers.rag.rag_repository.get_store_by_org", new_callable=AsyncMock, return_value=existing_store):

            response = client.post(
                "/api/v1/rag/stores",
                headers=headers,
                json={"display_name": "Test Store"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["store_id"] == "store-existing"


class TestListStoresEndpoint:
    """Tests for GET /stores endpoint."""

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

    def test_list_stores_success(self, client, headers):
        """Test listing stores for organization."""
        stores = [
            {
                "id": "store-1",
                "display_name": "store_one",
                "created_at": "2024-01-01T00:00:00Z",
                "active_documents_count": 10,
            }
        ]

        with patch("src.api.routers.rag.rag_repository.list_stores", new_callable=AsyncMock, return_value=stores):

            response = client.get("/api/v1/rag/stores", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["stores"]) == 1
            assert data["stores"][0]["store_id"] == "store-1"

    def test_list_stores_empty(self, client, headers):
        """Test listing stores when org has no stores."""
        with patch("src.api.routers.rag.rag_repository.list_stores", new_callable=AsyncMock, return_value=[]):

            response = client.get("/api/v1/rag/stores", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["stores"]) == 0


class TestGetStoreEndpoint:
    """Tests for GET /stores/{store_id} endpoint."""

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

    def test_get_store_success(self, client, headers):
        """Test getting a store by ID."""
        store_info = {
            "id": "store-123",
            "display_name": "test_store",
            "organization_id": TEST_ORG_ID,
            "gemini_store_id": "fileSearchStores/gemini-123",
            "created_at": "2024-01-01T00:00:00Z",
            "active_documents_count": 5,
        }

        with patch("src.api.routers.rag.validate_store_ownership", new_callable=AsyncMock, return_value=store_info):

            response = client.get("/api/v1/rag/stores/store-123", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # Response model is CreateStoreResponse which has store_id field
            assert data["store_id"] == "store-123"

    def test_get_store_not_found(self, client, headers):
        """Test 404 when store doesn't exist."""
        from fastapi import HTTPException

        with patch("src.api.routers.rag.validate_store_ownership", new_callable=AsyncMock, side_effect=HTTPException(status_code=404, detail="Store not found")):

            response = client.get("/api/v1/rag/stores/nonexistent", headers=headers)

            assert response.status_code == 404


class TestDeleteStoreEndpoint:
    """Tests for DELETE /stores/{store_id} endpoint."""

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

    def test_delete_store_success(self, client, headers):
        """Test deleting a store."""
        store_info = {
            "id": "store-123",
            "display_name": "test_store",
            "organization_id": TEST_ORG_ID,
            "gemini_store_id": "fileSearchStores/gemini-123",
        }

        with patch("src.api.routers.rag.validate_store_ownership", new_callable=AsyncMock, return_value=store_info), \
             patch("src.rag.gemini_file_store.delete_store") as mock_delete, \
             patch("src.api.routers.rag.rag_repository.delete_store", new_callable=AsyncMock, return_value=True):

            response = client.delete("/api/v1/rag/stores/store-123", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestListStoreFilesEndpoint:
    """Tests for GET /stores/{store_id}/files endpoint."""

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

    def test_list_store_files_success(self, client, headers):
        """Test listing files in a store."""
        store_info = {
            "id": "store-123",
            "display_name": "test_store",
            "organization_id": TEST_ORG_ID,
            "gemini_store_id": "fileSearchStores/gemini-123",
        }

        mock_files = [
            MagicMock(name="file1.md", display_name="file1.md"),
            MagicMock(name="file2.md", display_name="file2.md"),
        ]

        with patch("src.api.routers.rag.validate_store_ownership", new_callable=AsyncMock, return_value=store_info), \
             patch("src.rag.gemini_file_store.list_documents", return_value=mock_files):

            response = client.get("/api/v1/rag/stores/store-123/files", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "files" in data


class TestCreateFolderEndpoint:
    """Tests for POST /folders endpoint."""

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

    def test_create_folder_success(self, client, headers):
        """Test creating a new folder."""
        store_info = {
            "id": "store-123",
            "display_name": "test_store",
        }

        folder_data = {
            "id": "folder-new",
            "folder_name": "Invoices",
            "store_id": "store-123",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
            "organization_id": TEST_ORG_ID,
        }

        with patch("src.api.routers.rag.rag_repository.get_store_by_org", new_callable=AsyncMock, return_value=store_info), \
             patch("src.api.routers.rag.rag_repository.get_folder_by_name", new_callable=AsyncMock, return_value=None), \
             patch("src.api.routers.rag.rag_repository.create_folder", new_callable=AsyncMock, return_value=folder_data):

            response = client.post(
                "/api/v1/rag/folders",
                headers=headers,
                json={"folder_name": "Invoices"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # CreateFolderResponse has folder object with folder_id
            assert data["folder"]["folder_id"] == "folder-new"

    def test_create_folder_duplicate_name(self, client, headers):
        """Test error when folder name already exists."""
        store_info = {"id": "store-123"}
        existing_folder = {"id": "folder-existing", "name": "Invoices"}

        with patch("src.api.routers.rag.rag_repository.get_store_by_org", new_callable=AsyncMock, return_value=store_info), \
             patch("src.api.routers.rag.rag_repository.get_folder_by_name", new_callable=AsyncMock, return_value=existing_folder):

            response = client.post(
                "/api/v1/rag/folders",
                headers=headers,
                json={"folder_name": "Invoices"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "already exists" in data["error"]

    def test_create_folder_no_store(self, client, headers):
        """Test error when org has no store."""
        with patch("src.api.routers.rag.rag_repository.get_store_by_org", new_callable=AsyncMock, return_value=None):

            response = client.post(
                "/api/v1/rag/folders",
                headers=headers,
                json={"folder_name": "Invoices"}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is False
            assert "store" in data["error"].lower()


class TestListFoldersEndpoint:
    """Tests for GET /folders endpoint."""

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

    def test_list_folders_success(self, client, headers):
        """Test listing folders for organization."""
        folders = [
            {
                "id": "folder-1",
                "folder_name": "Invoices",
                "store_id": "store-123",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "organization_id": TEST_ORG_ID,
            },
            {
                "id": "folder-2",
                "folder_name": "Reports",
                "store_id": "store-123",
                "created_at": "2024-01-02T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
                "organization_id": TEST_ORG_ID,
            },
        ]

        with patch("src.api.routers.rag.rag_repository.list_folders", new_callable=AsyncMock, return_value=folders):

            response = client.get("/api/v1/rag/folders", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["folders"]) == 2


class TestGetFolderEndpoint:
    """Tests for GET /folders/{folder_id} endpoint."""

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

    def test_get_folder_success(self, client, headers):
        """Test getting a folder by ID."""
        folder_info = {
            "id": "folder-123",
            "folder_name": "Invoices",
            "organization_id": TEST_ORG_ID,
            "store_id": "store-123",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z",
        }

        with patch("src.api.routers.rag.validate_folder_ownership", new_callable=AsyncMock, return_value=folder_info):

            response = client.get("/api/v1/rag/folders/folder-123", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            # GetFolderResponse has folder object with folder_id
            assert data["folder"]["folder_id"] == "folder-123"

    def test_get_folder_not_found(self, client, headers):
        """Test 404 when folder doesn't exist."""
        from fastapi import HTTPException

        with patch("src.api.routers.rag.validate_folder_ownership", new_callable=AsyncMock, side_effect=HTTPException(status_code=404, detail="Folder not found")):

            response = client.get("/api/v1/rag/folders/nonexistent", headers=headers)

            assert response.status_code == 404


class TestDeleteFolderEndpoint:
    """Tests for DELETE /folders/{folder_id} endpoint."""

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

    def test_delete_folder_success(self, client, headers):
        """Test deleting a folder."""
        folder_info = {
            "id": "folder-123",
            "folder_name": "Invoices",
            "organization_id": TEST_ORG_ID,
            "store_id": "store-123",
            "document_count": 0,
            "total_size_bytes": 0,
        }

        with patch("src.api.routers.rag.validate_folder_ownership", new_callable=AsyncMock, return_value=folder_info), \
             patch("src.api.routers.rag.rag_repository.has_subfolders", new_callable=AsyncMock, return_value=False), \
             patch("src.api.routers.rag.rag_repository.delete_folder", new_callable=AsyncMock, return_value={"success": True, "document_count": 0}):

            response = client.delete("/api/v1/rag/folders/folder-123", headers=headers)

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

    def test_delete_folder_not_found(self, client, headers):
        """Test 404 when folder doesn't exist."""
        from fastapi import HTTPException

        with patch("src.api.routers.rag.validate_folder_ownership", new_callable=AsyncMock, side_effect=HTTPException(status_code=404, detail="Folder not found")):

            response = client.delete("/api/v1/rag/folders/nonexistent", headers=headers)

            assert response.status_code == 404
