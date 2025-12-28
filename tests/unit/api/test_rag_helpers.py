"""Unit tests for RAG helper functions."""

import pytest
import hashlib
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import HTTPException


class TestValidateStoreOwnership:
    """Tests for validate_store_ownership function."""

    @pytest.mark.asyncio
    async def test_validate_store_ownership_success(self):
        """Test successful store ownership validation."""
        from src.api.routers.rag_helpers import validate_store_ownership

        mock_store_info = {
            "id": "store-123",
            "organization_id": "org-123",
            "display_name": "test_store",
            "gemini_store_id": "fileSearchStores/abc",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_store_by_id = AsyncMock(return_value=mock_store_info)

            result = await validate_store_ownership("store-123", "org-123")

            assert result == mock_store_info
            mock_repo.get_store_by_id.assert_called_once_with("store-123")

    @pytest.mark.asyncio
    async def test_validate_store_ownership_not_found(self):
        """Test 404 when store doesn't exist."""
        from src.api.routers.rag_helpers import validate_store_ownership

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_store_by_id = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await validate_store_ownership("nonexistent-store", "org-123")

            assert exc_info.value.status_code == 404
            assert "Store not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_validate_store_ownership_wrong_org(self):
        """Test 403 when store belongs to another organization."""
        from src.api.routers.rag_helpers import validate_store_ownership

        mock_store_info = {
            "id": "store-123",
            "organization_id": "other-org",  # Different org
            "display_name": "test_store",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_store_by_id = AsyncMock(return_value=mock_store_info)

            with pytest.raises(HTTPException) as exc_info:
                await validate_store_ownership("store-123", "org-123")

            assert exc_info.value.status_code == 403
            assert "belongs to another organization" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_validate_store_ownership_no_org_in_store(self):
        """Test that store without org_id passes validation."""
        from src.api.routers.rag_helpers import validate_store_ownership

        mock_store_info = {
            "id": "store-123",
            "organization_id": None,  # No org set
            "display_name": "test_store",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_store_by_id = AsyncMock(return_value=mock_store_info)

            result = await validate_store_ownership("store-123", "org-123")

            # Should pass because store has no org restriction
            assert result == mock_store_info


class TestValidateFolderOwnership:
    """Tests for validate_folder_ownership function."""

    @pytest.mark.asyncio
    async def test_validate_folder_ownership_success(self):
        """Test successful folder ownership validation."""
        from src.api.routers.rag_helpers import validate_folder_ownership

        mock_folder_info = {
            "id": "folder-123",
            "organization_id": "org-123",
            "name": "Test Folder",
            "store_id": "store-123",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_folder_by_id = AsyncMock(return_value=mock_folder_info)

            result = await validate_folder_ownership("folder-123", "org-123")

            assert result == mock_folder_info
            mock_repo.get_folder_by_id.assert_called_once_with("folder-123")

    @pytest.mark.asyncio
    async def test_validate_folder_ownership_not_found(self):
        """Test 404 when folder doesn't exist."""
        from src.api.routers.rag_helpers import validate_folder_ownership

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_folder_by_id = AsyncMock(return_value=None)

            with pytest.raises(HTTPException) as exc_info:
                await validate_folder_ownership("nonexistent-folder", "org-123")

            assert exc_info.value.status_code == 404
            assert "Folder not found" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_validate_folder_ownership_wrong_org(self):
        """Test 403 when folder belongs to another organization."""
        from src.api.routers.rag_helpers import validate_folder_ownership

        mock_folder_info = {
            "id": "folder-123",
            "organization_id": "other-org",  # Different org
            "name": "Test Folder",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_folder_by_id = AsyncMock(return_value=mock_folder_info)

            with pytest.raises(HTTPException) as exc_info:
                await validate_folder_ownership("folder-123", "org-123")

            assert exc_info.value.status_code == 403
            assert "belongs to another organization" in exc_info.value.detail


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_compute_content_hash_success(self, tmp_path):
        """Test successful hash computation for local file."""
        from src.api.routers.rag_helpers import compute_content_hash

        # Create test file with known content
        test_content = b"Hello, World!"
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(test_content)

        result = compute_content_hash(str(test_file))

        expected_hash = hashlib.sha256(test_content).hexdigest()
        assert result == expected_hash

    def test_compute_content_hash_consistent(self, tmp_path):
        """Test that same content produces same hash."""
        from src.api.routers.rag_helpers import compute_content_hash

        content = b"Consistent content for hashing"

        file1 = tmp_path / "file1.txt"
        file1.write_bytes(content)

        file2 = tmp_path / "file2.txt"
        file2.write_bytes(content)

        hash1 = compute_content_hash(str(file1))
        hash2 = compute_content_hash(str(file2))

        assert hash1 == hash2

    def test_compute_content_hash_different_content(self, tmp_path):
        """Test that different content produces different hash."""
        from src.api.routers.rag_helpers import compute_content_hash

        file1 = tmp_path / "file1.txt"
        file1.write_bytes(b"Content A")

        file2 = tmp_path / "file2.txt"
        file2.write_bytes(b"Content B")

        hash1 = compute_content_hash(str(file1))
        hash2 = compute_content_hash(str(file2))

        assert hash1 != hash2


class TestComputeGcsContentHash:
    """Tests for compute_gcs_content_hash function."""

    @pytest.mark.asyncio
    async def test_compute_gcs_content_hash_success(self):
        """Test successful hash computation for GCS file."""
        from src.api.routers.rag_helpers import compute_gcs_content_hash

        test_content = "Hello from GCS!"

        mock_storage = MagicMock()
        mock_storage.read = AsyncMock(return_value=test_content)

        with patch("src.storage.get_storage", return_value=mock_storage):
            result = await compute_gcs_content_hash("gs://bucket/path/file.txt")

            expected_hash = hashlib.sha256(test_content.encode('utf-8')).hexdigest()
            assert result == expected_hash

    @pytest.mark.asyncio
    async def test_compute_gcs_content_hash_bytes(self):
        """Test hash computation for bytes content from GCS."""
        from src.api.routers.rag_helpers import compute_gcs_content_hash

        test_content = b"Binary content from GCS"

        mock_storage = MagicMock()
        mock_storage.read = AsyncMock(return_value=test_content)

        with patch("src.storage.get_storage", return_value=mock_storage):
            result = await compute_gcs_content_hash("gs://bucket/path/file.bin")

            expected_hash = hashlib.sha256(test_content).hexdigest()
            assert result == expected_hash

    @pytest.mark.asyncio
    async def test_compute_gcs_content_hash_file_not_found(self):
        """Test None returned when GCS file not found."""
        from src.api.routers.rag_helpers import compute_gcs_content_hash

        mock_storage = MagicMock()
        mock_storage.read = AsyncMock(return_value=None)

        with patch("src.storage.get_storage", return_value=mock_storage):
            result = await compute_gcs_content_hash("gs://bucket/nonexistent.txt")

            assert result is None

    @pytest.mark.asyncio
    async def test_compute_gcs_content_hash_error(self):
        """Test None returned on GCS read error."""
        from src.api.routers.rag_helpers import compute_gcs_content_hash

        mock_storage = MagicMock()
        mock_storage.read = AsyncMock(side_effect=Exception("GCS error"))

        with patch("src.storage.get_storage", return_value=mock_storage):
            result = await compute_gcs_content_hash("gs://bucket/error.txt")

            assert result is None


class TestGetOrCreateOrgStore:
    """Tests for get_or_create_org_store function."""

    @pytest.mark.asyncio
    async def test_returns_existing_store_from_db(self):
        """Test returning existing store from database."""
        from src.api.routers.rag_helpers import get_or_create_org_store

        existing_store = {
            "id": "store-123",
            "gemini_store_id": "fileSearchStores/abc",
            "display_name": "acme_corp_file_search_store",
            "organization_id": "org-123",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo:
            mock_repo.get_store_by_display_name = AsyncMock(return_value=existing_store)

            result = await get_or_create_org_store("org-123", "ACME Corp")

            assert result == existing_store
            mock_repo.get_store_by_display_name.assert_called_once()

    @pytest.mark.asyncio
    async def test_creates_new_store_when_not_in_db(self):
        """Test creating new store when not in database."""
        from src.api.routers.rag_helpers import get_or_create_org_store

        mock_gemini_store = MagicMock()
        mock_gemini_store.name = "fileSearchStores/new-store"

        new_store_data = {
            "id": "store-new",
            "gemini_store_id": "fileSearchStores/new-store",
            "display_name": "acme_corp_file_search_store",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo, \
             patch("src.rag.gemini_file_store.generate_store_display_name", return_value="acme_corp_file_search_store"), \
             patch("src.rag.gemini_file_store.get_or_create_store_by_org_name", return_value=(mock_gemini_store, True)):

            mock_repo.get_store_by_display_name = AsyncMock(return_value=None)
            mock_repo.create_store = AsyncMock(return_value=new_store_data)

            result = await get_or_create_org_store("org-123", "ACME Corp")

            assert result == new_store_data
            mock_repo.create_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_links_existing_gemini_store_to_db(self):
        """Test linking existing Gemini store to database."""
        from src.api.routers.rag_helpers import get_or_create_org_store

        mock_gemini_store = MagicMock()
        mock_gemini_store.name = "fileSearchStores/existing"

        linked_store_data = {
            "id": "store-linked",
            "gemini_store_id": "fileSearchStores/existing",
            "display_name": "acme_corp_file_search_store",
        }

        with patch("src.api.routers.rag_helpers.rag_repository") as mock_repo, \
             patch("src.rag.gemini_file_store.generate_store_display_name", return_value="acme_corp_file_search_store"), \
             patch("src.rag.gemini_file_store.get_or_create_store_by_org_name", return_value=(mock_gemini_store, False)):

            mock_repo.get_store_by_display_name = AsyncMock(return_value=None)
            mock_repo.get_or_create_store = AsyncMock(return_value=linked_store_data)

            result = await get_or_create_org_store("org-123", "ACME Corp")

            assert result == linked_store_data
            mock_repo.get_or_create_store.assert_called_once()
