"""Unit tests for webhook handler."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from src.bulk.webhook_handler import (
    handle_document_uploaded,
    _parse_bulk_path,
    _is_allowed_file,
    _find_or_create_job,
    _should_auto_start_job,
    _trigger_job_start,
    validate_webhook_request,
    ALLOWED_EXTENSIONS,
    BULK_PATH_PATTERN,
)


class TestParseBulkPath:
    """Tests for _parse_bulk_path helper function."""

    def test_parse_valid_path(self):
        """Test parsing a valid bulk path."""
        org_id, folder_name, filename = _parse_bulk_path("org123/bulk/invoices/doc.pdf")
        assert org_id == "org123"
        assert folder_name == "invoices"
        assert filename == "doc.pdf"

    def test_parse_valid_path_with_dashes(self):
        """Test parsing path with dashes in names."""
        org_id, folder_name, filename = _parse_bulk_path("my-org/bulk/my-folder/my-doc.pdf")
        assert org_id == "my-org"
        assert folder_name == "my-folder"
        assert filename == "my-doc.pdf"

    def test_parse_valid_path_with_underscores(self):
        """Test parsing path with underscores in names."""
        org_id, folder_name, filename = _parse_bulk_path("org_1/bulk/folder_2/doc_3.pdf")
        assert org_id == "org_1"
        assert folder_name == "folder_2"
        assert filename == "doc_3.pdf"

    def test_parse_valid_path_with_numbers(self):
        """Test parsing path with numbers in names."""
        org_id, folder_name, filename = _parse_bulk_path("org123/bulk/folder456/doc789.pdf")
        assert org_id == "org123"
        assert folder_name == "folder456"
        assert filename == "doc789.pdf"

    def test_parse_invalid_path_no_bulk(self):
        """Test error when path doesn't contain 'bulk'."""
        with pytest.raises(ValueError) as exc_info:
            _parse_bulk_path("org123/other/folder/doc.pdf")
        assert "Invalid bulk folder path" in str(exc_info.value)

    def test_parse_invalid_path_wrong_structure(self):
        """Test error for wrong path structure."""
        with pytest.raises(ValueError) as exc_info:
            _parse_bulk_path("org123/bulk")
        assert "Invalid bulk folder path" in str(exc_info.value)

    def test_parse_invalid_path_only_org(self):
        """Test error when only org is in path."""
        with pytest.raises(ValueError) as exc_info:
            _parse_bulk_path("org123")
        assert "Invalid bulk folder path" in str(exc_info.value)

    def test_parse_invalid_path_empty(self):
        """Test error for empty path."""
        with pytest.raises(ValueError) as exc_info:
            _parse_bulk_path("")
        assert "Invalid bulk folder path" in str(exc_info.value)

    def test_parse_invalid_path_nested_folder(self):
        """Test error for nested folders (not supported)."""
        with pytest.raises(ValueError) as exc_info:
            _parse_bulk_path("org123/bulk/folder/subfolder/doc.pdf")
        assert "Invalid bulk folder path" in str(exc_info.value)


class TestIsAllowedFile:
    """Tests for _is_allowed_file helper function."""

    def test_allowed_pdf(self):
        """Test PDF is allowed."""
        assert _is_allowed_file("document.pdf") is True

    def test_allowed_docx(self):
        """Test DOCX is allowed."""
        assert _is_allowed_file("document.docx") is True

    def test_allowed_doc(self):
        """Test DOC is allowed."""
        assert _is_allowed_file("document.doc") is True

    def test_allowed_pptx(self):
        """Test PPTX is allowed."""
        assert _is_allowed_file("presentation.pptx") is True

    def test_allowed_xlsx(self):
        """Test XLSX is allowed."""
        assert _is_allowed_file("spreadsheet.xlsx") is True

    def test_allowed_txt(self):
        """Test TXT is allowed."""
        assert _is_allowed_file("readme.txt") is True

    def test_allowed_md(self):
        """Test MD (markdown) is allowed."""
        assert _is_allowed_file("readme.md") is True

    def test_allowed_csv(self):
        """Test CSV is allowed."""
        assert _is_allowed_file("data.csv") is True

    def test_allowed_png(self):
        """Test PNG is allowed."""
        assert _is_allowed_file("image.png") is True

    def test_allowed_jpg(self):
        """Test JPG is allowed."""
        assert _is_allowed_file("photo.jpg") is True

    def test_allowed_jpeg(self):
        """Test JPEG is allowed."""
        assert _is_allowed_file("photo.jpeg") is True

    def test_not_allowed_exe(self):
        """Test EXE is not allowed."""
        assert _is_allowed_file("program.exe") is False

    def test_not_allowed_zip(self):
        """Test ZIP is not allowed."""
        assert _is_allowed_file("archive.zip") is False

    def test_not_allowed_sh(self):
        """Test shell scripts are not allowed."""
        assert _is_allowed_file("script.sh") is False

    def test_not_allowed_py(self):
        """Test Python files are not allowed."""
        assert _is_allowed_file("script.py") is False

    def test_empty_filename(self):
        """Test empty filename returns False."""
        assert _is_allowed_file("") is False

    def test_no_extension(self):
        """Test filename without extension returns False."""
        assert _is_allowed_file("noextension") is False

    def test_hidden_file(self):
        """Test hidden file with valid extension is checked."""
        # Hidden files like .hidden.pdf should not be allowed based on file extension
        # But the function checks extension, not hidden status
        assert _is_allowed_file(".hidden.pdf") is True

    def test_case_insensitive_extension(self):
        """Test extension matching is case insensitive."""
        assert _is_allowed_file("DOCUMENT.PDF") is True
        assert _is_allowed_file("Document.Pdf") is True


class TestValidateWebhookRequest:
    """Tests for validate_webhook_request function."""

    @pytest.mark.asyncio
    async def test_validate_success(self):
        """Test successful validation."""
        is_valid, error = await validate_webhook_request(
            bucket="test-bucket",
            file_path="org123/bulk/folder/doc.pdf",
        )
        assert is_valid is True
        assert error == ""

    @pytest.mark.asyncio
    async def test_validate_wrong_bucket(self):
        """Test validation fails for wrong bucket."""
        is_valid, error = await validate_webhook_request(
            bucket="wrong-bucket",
            file_path="org123/bulk/folder/doc.pdf",
            expected_bucket="correct-bucket",
        )
        assert is_valid is False
        assert "Unexpected bucket" in error

    @pytest.mark.asyncio
    async def test_validate_not_bulk_path(self):
        """Test validation fails when not a bulk path."""
        is_valid, error = await validate_webhook_request(
            bucket="test-bucket",
            file_path="org123/other/folder/doc.pdf",
        )
        assert is_valid is False
        assert "Not a bulk folder path" in error

    @pytest.mark.asyncio
    async def test_validate_invalid_format(self):
        """Test validation fails for invalid path format."""
        is_valid, error = await validate_webhook_request(
            bucket="test-bucket",
            file_path="org123/bulk/folder/subfolder/doc.pdf",
        )
        assert is_valid is False
        assert "Invalid bulk folder path" in error

    @pytest.mark.asyncio
    async def test_validate_no_expected_bucket(self):
        """Test validation passes when no expected bucket specified."""
        is_valid, error = await validate_webhook_request(
            bucket="any-bucket",
            file_path="org123/bulk/folder/doc.pdf",
            expected_bucket=None,
        )
        assert is_valid is True


class TestFindOrCreateJob:
    """Tests for _find_or_create_job helper function."""

    @pytest.mark.asyncio
    async def test_find_existing_job(self, sample_bulk_job_dict):
        """Test finding an existing pending job."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo:
            mock_repo.find_active_job_for_folder = AsyncMock(return_value=sample_bulk_job_dict)

            job = await _find_or_create_job(
                org_id="test-org",
                folder_name="test-folder",
                bucket="test-bucket",
            )

            assert job is not None
            assert job["id"] == "job-123"
            mock_repo.find_active_job_for_folder.assert_called_once_with("test-org", "test-folder")
            mock_repo.create_bulk_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_new_job(self):
        """Test creating a new job when none exists."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo:
            mock_repo.find_active_job_for_folder = AsyncMock(return_value=None)
            mock_repo.create_bulk_job = AsyncMock(return_value={
                "id": "new-job-123",
                "folder_name": "test-folder",
            })

            job = await _find_or_create_job(
                org_id="test-org",
                folder_name="test-folder",
                bucket="test-bucket",
            )

            assert job is not None
            assert job["id"] == "new-job-123"
            mock_repo.find_active_job_for_folder.assert_called_once()
            mock_repo.create_bulk_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_job_failure(self):
        """Test handling job creation failure."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo:
            mock_repo.find_active_job_for_folder = AsyncMock(return_value=None)
            mock_repo.create_bulk_job = AsyncMock(side_effect=Exception("DB error"))

            job = await _find_or_create_job(
                org_id="test-org",
                folder_name="test-folder",
                bucket="test-bucket",
            )

            assert job is None


class TestShouldAutoStartJob:
    """Tests for _should_auto_start_job helper function."""

    @pytest.mark.asyncio
    async def test_auto_start_min_documents_not_reached(self):
        """Test no auto-start when min documents not reached."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_config.return_value.auto_start_min_documents = 5
            mock_repo.get_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.count_documents_in_job = AsyncMock(return_value=2)

            result = await _should_auto_start_job("job-123", mock_config.return_value)
            assert result is False

    @pytest.mark.asyncio
    async def test_auto_start_delay_not_passed(self):
        """Test no auto-start when delay has not passed."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_config.return_value.auto_start_min_documents = 1
            mock_config.return_value.auto_start_delay_seconds = 60
            mock_repo.get_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.count_documents_in_job = AsyncMock(return_value=5)
            # Last doc was uploaded just now
            mock_repo.get_latest_document_in_job = AsyncMock(return_value={
                "created_at": datetime.utcnow(),
            })

            result = await _should_auto_start_job("job-123", mock_config.return_value)
            assert result is False

    @pytest.mark.asyncio
    async def test_auto_start_delay_passed(self):
        """Test auto-start when delay has passed."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_config.return_value.auto_start_min_documents = 1
            mock_config.return_value.auto_start_delay_seconds = 60
            mock_repo.get_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.count_documents_in_job = AsyncMock(return_value=5)
            # Last doc was uploaded 2 minutes ago
            mock_repo.get_latest_document_in_job = AsyncMock(return_value={
                "created_at": datetime.utcnow() - timedelta(seconds=120),
            })

            result = await _should_auto_start_job("job-123", mock_config.return_value)
            assert result is True

    @pytest.mark.asyncio
    async def test_auto_start_no_delay_configured(self):
        """Test auto-start immediately when no delay configured."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_config.return_value.auto_start_min_documents = 1
            mock_config.return_value.auto_start_delay_seconds = 0
            mock_repo.get_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.count_documents_in_job = AsyncMock(return_value=5)

            result = await _should_auto_start_job("job-123", mock_config.return_value)
            assert result is True

    @pytest.mark.asyncio
    async def test_auto_start_job_not_found(self):
        """Test no auto-start when job not found."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_repo.get_bulk_job = AsyncMock(return_value=None)

            result = await _should_auto_start_job("job-123", mock_config.return_value)
            assert result is False


class TestTriggerJobStart:
    """Tests for _trigger_job_start helper function."""

    @pytest.mark.asyncio
    async def test_trigger_job_start(self):
        """Test triggering job start via queue."""
        with patch("src.bulk.queue.get_bulk_queue") as mock_get_queue:
            mock_queue = MagicMock()
            mock_get_queue.return_value = mock_queue

            await _trigger_job_start("job-123")

            mock_queue.enqueue.assert_called_once()
            event = mock_queue.enqueue.call_args[0][0]
            assert event.job_id == "job-123"
            assert event.action == "start"


class TestHandleDocumentUploaded:
    """Tests for handle_document_uploaded main function."""

    @pytest.mark.asyncio
    async def test_handle_upload_success(self, sample_bulk_job_dict, sample_document_item_dict):
        """Test successful upload handling."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_config.return_value.max_documents_per_folder = 10
            mock_config.return_value.auto_start_enabled = False

            mock_repo.find_active_job_for_folder = AsyncMock(return_value=sample_bulk_job_dict)
            mock_repo.get_document_item_by_path = AsyncMock(return_value=None)
            mock_repo.count_documents_in_job = AsyncMock(return_value=2)
            mock_repo.create_document_item = AsyncMock(return_value=sample_document_item_dict)
            mock_repo.increment_total_documents = AsyncMock()

            result = await handle_document_uploaded(
                bucket="test-bucket",
                file_path="test-org/bulk/test-folder/test.pdf",
                file_size=1024,
            )

            assert result["success"] is True
            assert result["job_id"] == "job-123"
            assert result["document_id"] == "doc-123"
            assert result["action"] == "added"

    @pytest.mark.asyncio
    async def test_handle_upload_unsupported_file(self):
        """Test skipping unsupported file types."""
        result = await handle_document_uploaded(
            bucket="test-bucket",
            file_path="test-org/bulk/test-folder/script.exe",
            file_size=1024,
        )

        assert result["success"] is True
        assert result["action"] == "skipped"
        assert "not supported" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_upload_document_exists(self, sample_bulk_job_dict, sample_document_item_dict):
        """Test response when document already exists."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_config.return_value.max_documents_per_folder = 10

            mock_repo.find_active_job_for_folder = AsyncMock(return_value=sample_bulk_job_dict)
            mock_repo.get_document_item_by_path = AsyncMock(return_value=sample_document_item_dict)

            result = await handle_document_uploaded(
                bucket="test-bucket",
                file_path="test-org/bulk/test-folder/test.pdf",
                file_size=1024,
            )

            assert result["success"] is True
            assert result["action"] == "exists"
            assert "already registered" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_upload_limit_exceeded(self, sample_bulk_job_dict):
        """Test response when document limit exceeded."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config") as mock_config:

            mock_config.return_value.max_documents_per_folder = 5

            mock_repo.find_active_job_for_folder = AsyncMock(return_value=sample_bulk_job_dict)
            mock_repo.get_document_item_by_path = AsyncMock(return_value=None)
            mock_repo.count_documents_in_job = AsyncMock(return_value=5)  # At limit

            result = await handle_document_uploaded(
                bucket="test-bucket",
                file_path="test-org/bulk/test-folder/test.pdf",
                file_size=1024,
            )

            assert result["success"] is False
            assert result["action"] == "limit_exceeded"
            assert "maximum document limit" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_upload_invalid_path(self):
        """Test error on invalid bulk path."""
        with pytest.raises(ValueError) as exc_info:
            await handle_document_uploaded(
                bucket="test-bucket",
                file_path="invalid/path/doc.pdf",
                file_size=1024,
            )
        assert "Invalid bulk folder path" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_upload_job_creation_failed(self, mock_bulk_config):
        """Test error when job creation fails."""
        with patch("src.bulk.webhook_handler.bulk_repository") as mock_repo, \
             patch("src.bulk.webhook_handler.get_bulk_config", return_value=mock_bulk_config):

            mock_repo.find_active_job_for_folder = AsyncMock(return_value=None)
            mock_repo.create_bulk_job = AsyncMock(side_effect=Exception("DB error"))

            result = await handle_document_uploaded(
                bucket="test-bucket",
                file_path="test-org/bulk/test-folder/test.pdf",
                file_size=1024,
            )

            assert result["success"] is False
            assert result["action"] == "error"


class TestAllowedExtensions:
    """Tests for ALLOWED_EXTENSIONS constant."""

    def test_pdf_in_allowed(self):
        """Test .pdf is in allowed extensions."""
        assert ".pdf" in ALLOWED_EXTENSIONS

    def test_docx_in_allowed(self):
        """Test .docx is in allowed extensions."""
        assert ".docx" in ALLOWED_EXTENSIONS

    def test_pptx_in_allowed(self):
        """Test .pptx is in allowed extensions."""
        assert ".pptx" in ALLOWED_EXTENSIONS

    def test_xlsx_in_allowed(self):
        """Test .xlsx is in allowed extensions."""
        assert ".xlsx" in ALLOWED_EXTENSIONS

    def test_png_in_allowed(self):
        """Test .png is in allowed extensions."""
        assert ".png" in ALLOWED_EXTENSIONS

    def test_exe_not_in_allowed(self):
        """Test .exe is not in allowed extensions."""
        assert ".exe" not in ALLOWED_EXTENSIONS


class TestBulkPathPattern:
    """Tests for BULK_PATH_PATTERN regex."""

    def test_pattern_matches_valid_path(self):
        """Test pattern matches valid bulk path."""
        match = BULK_PATH_PATTERN.match("org123/bulk/folder/file.pdf")
        assert match is not None
        assert match.groups() == ("org123", "folder", "file.pdf")

    def test_pattern_no_match_nested(self):
        """Test pattern doesn't match nested paths."""
        match = BULK_PATH_PATTERN.match("org123/bulk/folder/sub/file.pdf")
        assert match is None

    def test_pattern_no_match_no_bulk(self):
        """Test pattern doesn't match without 'bulk'."""
        match = BULK_PATH_PATTERN.match("org123/other/folder/file.pdf")
        assert match is None
