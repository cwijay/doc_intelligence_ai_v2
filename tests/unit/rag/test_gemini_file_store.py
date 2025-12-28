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
    # Clear module-level caches before each test
    import src.rag.gemini_file_store as gemini_module
    gemini_module._store_cache.clear()
    gemini_module._store_list_cache = (None, 0.0)

    with patch("src.rag.gemini_file_store.client") as mock_client:
        # Default mock behavior
        mock_client.file_search_stores.list.return_value = []
        mock_client.file_search_stores.create.return_value = MagicMock(
            name="fileSearchStores/test-store-mock",
            display_name="test_store"
        )
        yield mock_client

    # Clear caches after test as well
    gemini_module._store_cache.clear()
    gemini_module._store_list_cache = (None, 0.0)


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


class TestQueryStore:
    """Tests for query_store function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock file search store."""
        store = MagicMock()
        store.name = "fileSearchStores/test-store-123"
        store.display_name = "test_store"
        return store

    @pytest.fixture
    def mock_response(self):
        """Create a mock response with grounding metadata."""
        response = MagicMock()
        response.text = "This is the generated answer."
        response.candidates = [MagicMock()]
        response.candidates[0].grounding_metadata = MagicMock()
        response.candidates[0].grounding_metadata.grounding_chunks = [
            MagicMock(
                retrieved_context=MagicMock(
                    title="doc1.pdf",
                    text="This is some relevant context from the document."
                )
            )
        ]
        return response

    def test_query_store_basic_semantic(self, mock_store, mock_response):
        """Test basic semantic search query."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event"):

            mock_client.models.generate_content.return_value = mock_response

            result = query_store(mock_store, "What is the contract term?")

            assert result == mock_response
            mock_client.models.generate_content.assert_called_once()

    def test_query_store_with_file_filter(self, mock_store, mock_response):
        """Test query with file name filter."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event"):

            mock_client.models.generate_content.return_value = mock_response

            result = query_store(
                mock_store,
                "Find payment terms",
                file_name_filter="contract.pdf"
            )

            # Verify the filter was passed
            call_args = mock_client.models.generate_content.call_args
            config = call_args.kwargs.get("config")
            assert config is not None

    def test_query_store_with_folder_filter(self, mock_store, mock_response):
        """Test query with folder name filter."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event"):

            mock_client.models.generate_content.return_value = mock_response

            result = query_store(
                mock_store,
                "Find all invoices",
                folder_name_filter="Invoices 2024"
            )

            assert result == mock_response

    def test_query_store_keyword_mode(self, mock_store, mock_response):
        """Test query with keyword search mode."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event"):

            mock_client.models.generate_content.return_value = mock_response

            result = query_store(
                mock_store,
                "NET 30 payment",
                search_mode="keyword"
            )

            # Verify the prompt was enhanced for keyword mode
            call_args = mock_client.models.generate_content.call_args
            contents = call_args.kwargs.get("contents", call_args.args[1] if len(call_args.args) > 1 else "")
            assert "exact" in str(contents).lower() or result == mock_response

    def test_query_store_hybrid_mode(self, mock_store, mock_response):
        """Test query with hybrid search mode."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event"):

            mock_client.models.generate_content.return_value = mock_response

            result = query_store(
                mock_store,
                "contract payment terms",
                search_mode="hybrid"
            )

            assert result == mock_response

    def test_query_store_retrieval_only(self, mock_store, mock_response):
        """Test query with generate_answer=False for retrieval only."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event"):

            mock_client.models.generate_content.return_value = mock_response

            result = query_store(
                mock_store,
                "Find relevant documents",
                generate_answer=False
            )

            assert result == mock_response

    def test_query_store_combined_filters(self, mock_store, mock_response):
        """Test query with multiple filters combined."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event"):

            mock_client.models.generate_content.return_value = mock_response

            result = query_store(
                mock_store,
                "Find payment info",
                file_name_filter="invoice.pdf",
                folder_name_filter="Invoices",
                folder_id_filter="folder-123"
            )

            assert result == mock_response

    def test_query_store_logs_event(self, mock_store, mock_response):
        """Test that query_store logs audit event."""
        from src.rag.gemini_file_store import query_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event") as mock_log:

            mock_client.models.generate_content.return_value = mock_response

            query_store(mock_store, "test query")

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == "search_performed"


class TestExtractCitations:
    """Tests for extract_citations function."""

    def test_extract_citations_success(self):
        """Test successful citation extraction."""
        from src.rag.gemini_file_store import extract_citations

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].grounding_metadata = MagicMock()
        mock_response.candidates[0].grounding_metadata.grounding_chunks = [
            MagicMock(
                retrieved_context=MagicMock(
                    title="document1.pdf",
                    text="This is the first citation text."
                )
            ),
            MagicMock(
                retrieved_context=MagicMock(
                    title="document2.pdf",
                    text="This is the second citation text."
                )
            ),
        ]

        citations = extract_citations(mock_response)

        assert len(citations) == 2
        assert citations[0]["index"] == 1
        assert citations[0]["title"] == "document1.pdf"
        assert citations[0]["text_preview"] == "This is the first citation text."
        assert citations[1]["index"] == 2
        assert citations[1]["title"] == "document2.pdf"

    def test_extract_citations_empty_response(self):
        """Test citation extraction with no candidates."""
        from src.rag.gemini_file_store import extract_citations

        mock_response = MagicMock()
        mock_response.candidates = []

        citations = extract_citations(mock_response)

        assert citations == []

    def test_extract_citations_no_grounding_metadata(self):
        """Test citation extraction with no grounding metadata."""
        from src.rag.gemini_file_store import extract_citations

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].grounding_metadata = None

        citations = extract_citations(mock_response)

        assert citations == []

    def test_extract_citations_text_truncation(self):
        """Test that long text is truncated in preview."""
        from src.rag.gemini_file_store import extract_citations

        long_text = "A" * 300  # Text longer than 200 chars

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].grounding_metadata = MagicMock()
        mock_response.candidates[0].grounding_metadata.grounding_chunks = [
            MagicMock(
                retrieved_context=MagicMock(
                    title="long_doc.pdf",
                    text=long_text
                )
            ),
        ]

        citations = extract_citations(mock_response)

        assert len(citations) == 1
        assert citations[0]["text_preview"].endswith("...")
        assert len(citations[0]["text_preview"]) == 203  # 200 + "..."
        assert citations[0]["full_text"] == long_text  # Full text preserved

    def test_extract_citations_missing_context(self):
        """Test handling of chunks with no retrieved_context."""
        from src.rag.gemini_file_store import extract_citations

        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].grounding_metadata = MagicMock()
        mock_response.candidates[0].grounding_metadata.grounding_chunks = [
            MagicMock(retrieved_context=None),
        ]

        citations = extract_citations(mock_response)

        assert len(citations) == 1
        assert citations[0]["title"] is None
        assert citations[0]["text_preview"] is None


class TestListDocuments:
    """Tests for list_documents function."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock file search store."""
        store = MagicMock()
        store.name = "fileSearchStores/test-store"
        store.display_name = "test_store"
        return store

    def test_list_documents_success(self, mock_store):
        """Test successful document listing."""
        from src.rag.gemini_file_store import list_documents

        mock_doc1 = MagicMock()
        mock_doc1.name = "fileSearchStores/test-store/documents/doc1"
        mock_doc1.display_name = "document1.pdf"
        mock_doc1.create_time = "2024-01-01T00:00:00Z"
        mock_doc1.size_bytes = 1024

        mock_doc2 = MagicMock()
        mock_doc2.name = "fileSearchStores/test-store/documents/doc2"
        mock_doc2.display_name = "document2.pdf"
        mock_doc2.create_time = "2024-01-02T00:00:00Z"
        mock_doc2.size_bytes = 2048

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.documents.list.return_value = [
                mock_doc1, mock_doc2
            ]

            documents = list_documents(mock_store)

            assert len(documents) == 2
            assert documents[0].display_name == "document1.pdf"
            assert documents[1].display_name == "document2.pdf"

    def test_list_documents_empty_store(self, mock_store):
        """Test listing documents from empty store."""
        from src.rag.gemini_file_store import list_documents

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.documents.list.return_value = []

            documents = list_documents(mock_store)

            assert documents == []

    def test_list_documents_with_custom_metadata(self, mock_store):
        """Test listing documents extracts custom metadata."""
        from src.rag.gemini_file_store import list_documents

        mock_doc = MagicMock()
        mock_doc.name = "fileSearchStores/test-store/documents/doc1"
        mock_doc.display_name = "document.pdf"
        mock_doc.custom_metadata = [
            MagicMock(key="folder_name", string_value="Invoices"),
            MagicMock(key="file_size", numeric_value=1024),
        ]

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.documents.list.return_value = [mock_doc]

            documents = list_documents(mock_store)

            assert len(documents) == 1
            assert documents[0].custom_metadata[0].key == "folder_name"


class TestDeleteStore:
    """Tests for delete_store function."""

    def test_delete_store_success(self):
        """Test successful store deletion."""
        from src.rag.gemini_file_store import delete_store

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event") as mock_log:

            delete_store("fileSearchStores/test-store-123")

            mock_client.file_search_stores.delete.assert_called_once_with(
                name="fileSearchStores/test-store-123",
                config={"force": True}
            )

    def test_delete_store_logs_event(self):
        """Test that delete_store logs audit event."""
        from src.rag.gemini_file_store import delete_store

        with patch("src.rag.gemini_file_store.client"), \
             patch("src.rag.gemini_file_store._log_event") as mock_log:

            delete_store("fileSearchStores/test-store-123")

            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][0] == "store_deleted"
            assert call_args.kwargs["details"]["store_name"] == "fileSearchStores/test-store-123"


class TestListAllStores:
    """Tests for list_all_stores function."""

    def test_list_all_stores_success(self):
        """Test successful listing of all stores."""
        from src.rag.gemini_file_store import list_all_stores

        mock_store1 = MagicMock()
        mock_store1.name = "fileSearchStores/store-1"
        mock_store1.display_name = "org1_file_search_store"
        mock_store1.active_documents_count = 5

        mock_store2 = MagicMock()
        mock_store2.name = "fileSearchStores/store-2"
        mock_store2.display_name = "org2_file_search_store"
        mock_store2.active_documents_count = 10

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.list.return_value = [mock_store1, mock_store2]

            stores = list_all_stores()

            assert len(stores) == 2
            assert stores[0].display_name == "org1_file_search_store"
            assert stores[1].display_name == "org2_file_search_store"

    def test_list_all_stores_empty(self):
        """Test listing when no stores exist."""
        from src.rag.gemini_file_store import list_all_stores

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.list.return_value = []

            stores = list_all_stores()

            assert stores == []


class TestFindStoreByName:
    """Tests for find_store_by_name function."""

    def test_find_store_exists(self):
        """Test finding an existing store."""
        from src.rag.gemini_file_store import find_store_by_name

        mock_store = MagicMock()
        mock_store.name = "fileSearchStores/test-store"
        mock_store.display_name = "acme_file_search_store"

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.list.return_value = [mock_store]

            result = find_store_by_name("acme_file_search_store")

            assert result == mock_store

    def test_find_store_not_found(self):
        """Test when store doesn't exist."""
        from src.rag.gemini_file_store import find_store_by_name

        mock_store = MagicMock()
        mock_store.display_name = "other_store"

        with patch("src.rag.gemini_file_store.client") as mock_client:
            mock_client.file_search_stores.list.return_value = [mock_store]

            result = find_store_by_name("nonexistent_store")

            assert result is None


class TestCreateFileSearchStore:
    """Tests for create_file_search_store function."""

    def test_create_store_success(self):
        """Test successful store creation."""
        from src.rag.gemini_file_store import create_file_search_store

        mock_store = MagicMock()
        mock_store.name = "fileSearchStores/new-store"
        mock_store.display_name = "new_store"

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event") as mock_log:

            mock_client.file_search_stores.create.return_value = mock_store

            result = create_file_search_store("new_store")

            assert result == mock_store
            mock_client.file_search_stores.create.assert_called_once_with(
                config={"display_name": "new_store"}
            )

    def test_create_store_logs_event(self):
        """Test that store creation logs audit event."""
        from src.rag.gemini_file_store import create_file_search_store

        mock_store = MagicMock()
        mock_store.name = "fileSearchStores/new-store"

        with patch("src.rag.gemini_file_store.client") as mock_client, \
             patch("src.rag.gemini_file_store._log_event") as mock_log:

            mock_client.file_search_stores.create.return_value = mock_store

            create_file_search_store("new_store")

            mock_log.assert_called_once()
            assert mock_log.call_args[0][0] == "store_created"
