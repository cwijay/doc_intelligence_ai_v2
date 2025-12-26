"""Unit tests for Gemini File Store functionality."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


# Track stores created during tests for cleanup
@pytest.fixture(autouse=True)
def mock_gemini_client_global():
    """
    Mock the Gemini client globally to prevent real API calls.
    This ensures tests don't create real Gemini stores.
    """
    with patch("src.rag.gemini_file_store.client") as mock_client:
        # Default mock behavior
        mock_client.file_search_stores.list.return_value = []
        mock_client.file_search_stores.create.return_value = MagicMock(
            name="fileSearchStores/test-store-mock",
            display_name="test_store"
        )
        yield mock_client


class TestGenerateStoreDisplayName:
    """Tests for generate_store_display_name function."""

    def test_simple_org_name(self):
        """Test with simple org name."""
        from src.rag.gemini_file_store import generate_store_display_name

        result = generate_store_display_name("acme")
        assert result == "acme_file_search_store"

    def test_org_name_with_spaces(self):
        """Test org name with spaces is converted to underscores."""
        from src.rag.gemini_file_store import generate_store_display_name

        result = generate_store_display_name("ACME Corp")
        assert result == "acme_corp_file_search_store"

    def test_org_name_with_hyphens(self):
        """Test org name with hyphens is converted to underscores."""
        from src.rag.gemini_file_store import generate_store_display_name

        result = generate_store_display_name("acme-corp")
        assert result == "acme_corp_file_search_store"

    def test_org_name_mixed_case(self):
        """Test org name with mixed case is lowercased."""
        from src.rag.gemini_file_store import generate_store_display_name

        result = generate_store_display_name("ACME Corp Inc")
        assert result == "acme_corp_inc_file_search_store"

    def test_org_name_with_double_spaces(self):
        """Test org name with double spaces doesn't create double underscores."""
        from src.rag.gemini_file_store import generate_store_display_name

        result = generate_store_display_name("ACME  Corp")
        assert result == "acme_corp_file_search_store"
        assert "__" not in result

    def test_org_name_with_mixed_separators(self):
        """Test org name with mixed spaces and hyphens."""
        from src.rag.gemini_file_store import generate_store_display_name

        result = generate_store_display_name("ACME - Corp - Inc")
        assert result == "acme_corp_inc_file_search_store"
        assert "__" not in result

    def test_case_insensitivity(self):
        """Test that different cases produce same result."""
        from src.rag.gemini_file_store import generate_store_display_name

        result1 = generate_store_display_name("ACME Corp")
        result2 = generate_store_display_name("acme corp")
        result3 = generate_store_display_name("Acme Corp")

        assert result1 == result2 == result3 == "acme_corp_file_search_store"


class TestGetOrCreateStoreByOrgName:
    """Tests for get_or_create_store_by_org_name function."""

    @pytest.fixture
    def mock_store_object(self):
        """Create a mock Gemini store object."""
        mock_store = MagicMock()
        mock_store.name = "fileSearchStores/test-store-123"
        mock_store.display_name = "acme_corp_file_search_store"
        return mock_store

    def test_finds_existing_store(self, mock_store_object):
        """Test finding an existing store by org name."""
        from src.rag.gemini_file_store import get_or_create_store_by_org_name

        with patch("src.rag.gemini_file_store.find_store_by_name", return_value=mock_store_object):
            store, is_new = get_or_create_store_by_org_name("ACME Corp")

            assert store == mock_store_object
            assert is_new is False

    def test_creates_new_store_when_not_found(self, mock_store_object):
        """Test creating a new store when none exists."""
        from src.rag.gemini_file_store import get_or_create_store_by_org_name

        with patch("src.rag.gemini_file_store.find_store_by_name", return_value=None), \
             patch("src.rag.gemini_file_store.create_file_search_store", return_value=mock_store_object):

            store, is_new = get_or_create_store_by_org_name("ACME Corp")

            assert store == mock_store_object
            assert is_new is True


class TestUploadFileMetadata:
    """Tests for upload_file function with enhanced metadata."""

    @pytest.fixture
    def mock_gemini_store(self):
        """Create a mock Gemini store."""
        store = MagicMock()
        store.name = "fileSearchStores/test-store"
        return store

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary test file."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test Content\n\nThis is test content.")
        return str(test_file)

    def test_upload_with_enhanced_metadata(self, mock_gemini_store, temp_file):
        """Test upload_file includes enhanced metadata."""
        from src.rag.gemini_file_store import upload_file

        mock_operation = MagicMock()
        mock_operation.done = True

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.documents.list.return_value = []
            mock_client.file_search_stores.upload_to_file_search_store.return_value = mock_operation

            upload_file(
                mock_gemini_store,
                temp_file,
                organization_id="org-123",
                folder_id="folder-456",
                folder_name="Test Folder",
                org_name="ACME Corp",
                content_hash="abc123hash",
                original_gcs_path="gs://bucket/original/doc.pdf",
                parsed_gcs_path="gs://bucket/parsed/doc.md",
                original_file_extension=".pdf",
                original_file_size=1024,
                parse_date="2024-01-01T00:00:00Z",
                parser_version="llama_parse_v2.5",
            )

            # Verify upload was called
            mock_client.file_search_stores.upload_to_file_search_store.assert_called_once()

            # Get the config passed to upload
            call_args = mock_client.file_search_stores.upload_to_file_search_store.call_args
            config = call_args.kwargs.get("config", {})
            custom_metadata = config.get("custom_metadata", [])

            # Convert to dict for easier assertion
            metadata_dict = {m["key"]: m.get("string_value") or m.get("numeric_value") for m in custom_metadata}

            assert metadata_dict["organization_id"] == "org-123"
            assert metadata_dict["org_name"] == "ACME Corp"
            assert metadata_dict["folder_id"] == "folder-456"
            assert metadata_dict["folder_name"] == "Test Folder"
            assert metadata_dict["content_hash"] == "abc123hash"
            assert metadata_dict["original_gcs_path"] == "gs://bucket/original/doc.pdf"
            assert metadata_dict["parsed_gcs_path"] == "gs://bucket/parsed/doc.md"
            assert metadata_dict["original_file_extension"] == ".pdf"
            assert metadata_dict["original_file_size"] == 1024
            assert metadata_dict["parse_date"] == "2024-01-01T00:00:00Z"
            assert metadata_dict["parser_version"] == "llama_parse_v2.5"
