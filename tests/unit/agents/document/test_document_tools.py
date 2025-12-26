"""Unit tests for DocumentLoaderTool and ContentPersistTool."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock


class TestDocumentLoaderTool:
    """Tests for DocumentLoaderTool class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = MagicMock()
        storage.exists = AsyncMock(return_value=True)
        storage.read = AsyncMock(return_value="# Test Content")
        storage.get_uri = MagicMock(return_value="gs://bucket/prefix/parsed/test.md")
        return storage

    @pytest.fixture
    def mock_storage_config(self):
        """Create a mock storage config."""
        config = MagicMock()
        config.parsed_directory = "parsed"
        return config

    @pytest.fixture
    def document_loader(self):
        """Create a DocumentLoaderTool instance."""
        from src.agents.document.tools import DocumentLoaderTool
        return DocumentLoaderTool()

    def test_load_from_gcs_success(
        self, document_loader, mock_storage, mock_storage_config
    ):
        """Test loading document from GCS successfully."""
        mock_storage.bucket_name = "test-bucket"

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config):

            # Now requires parsed_file_path parameter
            result = document_loader._run(
                document_name="test.md",
                parsed_file_path="Acme corp/parsed/invoices/test.md"
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["content"] == "# Test Content"
            assert result_data["source_type"] == "parsed_gcs"

    def test_load_from_gcs_with_parsed_file_path(
        self, document_loader, mock_storage, mock_storage_config
    ):
        """Test that parsed_file_path is used directly for GCS lookup."""
        mock_storage.bucket_name = "test-bucket"

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config):

            document_loader._run(
                document_name="test.md",
                parsed_file_path="Acme corp/parsed/invoices/test.md"
            )

            # Should check for the exact parsed_file_path
            mock_storage.exists.assert_called()

    def test_load_from_gcs_not_found_falls_back(
        self, document_loader, mock_storage, mock_storage_config, temp_upload_dir
    ):
        """Test fallback to upload directory when GCS file not found."""
        mock_storage.exists = AsyncMock(return_value=False)

        # Create a file in upload directory
        test_file = temp_upload_dir / "test.md"
        test_file.write_text("# Local Content")

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config), \
             patch.object(document_loader.config, "upload_directory", str(temp_upload_dir)):

            result = document_loader._run("test.md")
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["source_type"] == "upload"

    def test_load_from_upload_dir(self, document_loader, temp_upload_dir):
        """Test loading from local upload directory."""
        # Create a test file
        test_file = temp_upload_dir / "local.md"
        test_file.write_text("# Local Document")

        # Mock GCS to return not found
        mock_storage = MagicMock()
        mock_storage.exists = AsyncMock(return_value=False)

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config"), \
             patch.object(document_loader.config, "upload_directory", str(temp_upload_dir)):

            result = document_loader._run("local.md")
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert "# Local Document" in result_data["content"]

    def test_load_document_not_found(self, document_loader, temp_upload_dir):
        """Test error when document not found anywhere."""
        mock_storage = MagicMock()
        mock_storage.exists = AsyncMock(return_value=False)

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config"), \
             patch.object(document_loader.config, "upload_directory", str(temp_upload_dir)):

            result = document_loader._run("nonexistent.md")
            result_data = json.loads(result)

            assert result_data["success"] is False
            assert "not found" in result_data["error"]


class TestContentPersistTool:
    """Tests for ContentPersistTool class."""

    @pytest.fixture
    def mock_storage(self):
        """Create a mock storage backend."""
        storage = MagicMock()
        storage.save = AsyncMock(return_value="gs://bucket/prefix/generated/test_generated.json")
        return storage

    @pytest.fixture
    def mock_storage_config(self):
        """Create a mock storage config."""
        config = MagicMock()
        config.generated_directory = "generated"
        return config

    @pytest.fixture
    def content_persist(self):
        """Create a ContentPersistTool instance."""
        from src.agents.document.tools import ContentPersistTool
        tool = ContentPersistTool()
        # Disable database persistence for unit tests
        tool.config.persist_to_database = False
        return tool

    def test_persist_to_gcs_success(
        self, content_persist, mock_storage, mock_storage_config
    ):
        """Test persisting content to GCS successfully."""
        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config):

            result = content_persist._run(
                document_name="test.md",
                summary="This is a summary."
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["file_saved"] is True
            assert "gs://" in result_data["output_file_path"]

    def test_persist_with_all_content(
        self, content_persist, mock_storage, mock_storage_config
    ):
        """Test persisting summary, FAQs, and questions to separate files."""
        faqs = json.dumps([{"question": "Q1", "answer": "A1"}])
        questions = json.dumps([{"question": "Q1", "expected_answer": "A1", "difficulty": "easy"}])

        # Mock returns different URIs for each content type
        mock_storage.save = AsyncMock(side_effect=[
            "gs://bucket/org/summary/folder/test.md",
            "gs://bucket/org/faqs/folder/test.json",
            "gs://bucket/org/questions/folder/test.json"
        ])

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config):

            result = content_persist._run(
                document_name="test.md",
                parsed_file_path="org/parsed/folder/test.md",
                summary="Summary",
                faqs=faqs,
                questions=questions
            )
            result_data = json.loads(result)

            assert result_data["success"] is True
            assert result_data["file_saved"] is True

            # Verify save was called 3 times (once for each content type)
            assert mock_storage.save.call_count == 3

            # Verify output_file_paths contains all three paths
            assert "output_file_paths" in result_data
            assert "summary" in result_data["output_file_paths"]
            assert "faqs" in result_data["output_file_paths"]
            assert "questions" in result_data["output_file_paths"]

    def test_persist_gcs_failure_logs_error(
        self, content_persist, mock_storage, mock_storage_config
    ):
        """Test that GCS failure is logged but doesn't crash."""
        mock_storage.save = AsyncMock(side_effect=Exception("GCS Error"))

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config):

            result = content_persist._run(
                document_name="test.md",
                summary="Summary"
            )
            result_data = json.loads(result)

            # Should still succeed overall but file_saved should be False
            assert result_data["success"] is True
            assert result_data["file_saved"] is False
            assert "file_error" in result_data

    def test_persist_generates_correct_filename(
        self, content_persist, mock_storage, mock_storage_config
    ):
        """Test that output filenames are correctly generated for each content type."""
        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config):

            content_persist._run(
                document_name="my_document.md",
                parsed_file_path="org/parsed/folder/my_document.md",
                summary="Summary"
            )

            save_call = mock_storage.save.call_args
            filename = save_call[0][1]  # Second positional arg is filename
            # Summary files are now saved as markdown with document base name
            assert filename == "my_document.md"

    def test_persist_generates_correct_paths_for_each_type(
        self, content_persist, mock_storage, mock_storage_config
    ):
        """Test that each content type gets saved to the correct directory."""
        faqs = json.dumps([{"question": "Q1", "answer": "A1"}])
        questions = json.dumps([{"question": "Q1", "expected_answer": "A1", "difficulty": "easy"}])

        # Track all save calls
        save_calls = []
        async def track_save(content, filename, directory="", use_prefix=True):
            save_calls.append({"filename": filename, "directory": directory})
            return f"gs://bucket/{directory}/{filename}"

        mock_storage.save = AsyncMock(side_effect=track_save)

        with patch("src.storage.get_storage", return_value=mock_storage), \
             patch("src.storage.get_storage_config", return_value=mock_storage_config):

            content_persist._run(
                document_name="doc.pdf",
                parsed_file_path="Acme corp/parsed/invoices/doc.md",
                summary="Summary",
                faqs=faqs,
                questions=questions
            )

            # Should have 3 save calls
            assert len(save_calls) == 3

            # Check directories are correct
            directories = [call["directory"] for call in save_calls]
            assert "Acme corp/summary/invoices" in directories
            assert "Acme corp/faqs/invoices" in directories
            assert "Acme corp/questions/invoices" in directories

            # Check filenames are correct
            filenames = [call["filename"] for call in save_calls]
            assert "doc.md" in filenames  # summary as markdown
            assert "doc.json" in filenames  # faqs and questions as json
