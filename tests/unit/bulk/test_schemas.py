"""Unit tests for bulk processing schemas."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.bulk.schemas import (
    BulkJobStatus,
    DocumentItemStatus,
    ProcessingOptions,
    BulkFolderInfo,
    SignedUrlInfo,
    DocumentItemInfo,
    BulkJobInfo,
    BulkJobEvent,
    WebhookPayload,
    DocumentProcessingEvent,
)


class TestBulkJobStatus:
    """Tests for BulkJobStatus enum."""

    def test_pending_value(self):
        """Test pending status value."""
        assert BulkJobStatus.PENDING.value == "pending"

    def test_processing_value(self):
        """Test processing status value."""
        assert BulkJobStatus.PROCESSING.value == "processing"

    def test_completed_value(self):
        """Test completed status value."""
        assert BulkJobStatus.COMPLETED.value == "completed"

    def test_partial_failure_value(self):
        """Test partial_failure status value."""
        assert BulkJobStatus.PARTIAL_FAILURE.value == "partial_failure"

    def test_failed_value(self):
        """Test failed status value."""
        assert BulkJobStatus.FAILED.value == "failed"

    def test_cancelled_value(self):
        """Test cancelled status value."""
        assert BulkJobStatus.CANCELLED.value == "cancelled"


class TestDocumentItemStatus:
    """Tests for DocumentItemStatus enum."""

    def test_pending_value(self):
        """Test pending status value."""
        assert DocumentItemStatus.PENDING.value == "pending"

    def test_parsing_value(self):
        """Test parsing status value."""
        assert DocumentItemStatus.PARSING.value == "parsing"

    def test_parsed_value(self):
        """Test parsed status value."""
        assert DocumentItemStatus.PARSED.value == "parsed"

    def test_indexing_value(self):
        """Test indexing status value."""
        assert DocumentItemStatus.INDEXING.value == "indexing"

    def test_indexed_value(self):
        """Test indexed status value."""
        assert DocumentItemStatus.INDEXED.value == "indexed"

    def test_generating_value(self):
        """Test generating status value."""
        assert DocumentItemStatus.GENERATING.value == "generating"

    def test_completed_value(self):
        """Test completed status value."""
        assert DocumentItemStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        """Test failed status value."""
        assert DocumentItemStatus.FAILED.value == "failed"

    def test_skipped_value(self):
        """Test skipped status value."""
        assert DocumentItemStatus.SKIPPED.value == "skipped"


class TestProcessingOptions:
    """Tests for ProcessingOptions schema."""

    def test_default_values(self):
        """Test default option values."""
        options = ProcessingOptions()
        assert options.generate_summary is True
        assert options.generate_faqs is True
        assert options.generate_questions is True
        assert options.num_faqs == 10
        assert options.num_questions == 10
        assert options.summary_max_words == 500

    def test_custom_values(self):
        """Test custom option values."""
        options = ProcessingOptions(
            generate_summary=False,
            generate_faqs=True,
            generate_questions=False,
            num_faqs=5,
            num_questions=20,
            summary_max_words=1000,
        )
        assert options.generate_summary is False
        assert options.num_faqs == 5
        assert options.num_questions == 20
        assert options.summary_max_words == 1000

    def test_validation_num_faqs_min(self):
        """Test num_faqs minimum validation (1)."""
        with pytest.raises(ValidationError):
            ProcessingOptions(num_faqs=0)

    def test_validation_num_faqs_max(self):
        """Test num_faqs maximum validation (50)."""
        with pytest.raises(ValidationError):
            ProcessingOptions(num_faqs=51)

    def test_validation_num_questions_min(self):
        """Test num_questions minimum validation (1)."""
        with pytest.raises(ValidationError):
            ProcessingOptions(num_questions=0)

    def test_validation_num_questions_max(self):
        """Test num_questions maximum validation (100)."""
        with pytest.raises(ValidationError):
            ProcessingOptions(num_questions=101)

    def test_validation_summary_words_min(self):
        """Test summary_max_words minimum validation (50)."""
        with pytest.raises(ValidationError):
            ProcessingOptions(summary_max_words=49)

    def test_validation_summary_words_max(self):
        """Test summary_max_words maximum validation (2000)."""
        with pytest.raises(ValidationError):
            ProcessingOptions(summary_max_words=2001)

    def test_valid_boundary_num_faqs(self):
        """Test valid boundary values for num_faqs."""
        options1 = ProcessingOptions(num_faqs=1)
        assert options1.num_faqs == 1
        options50 = ProcessingOptions(num_faqs=50)
        assert options50.num_faqs == 50

    def test_valid_boundary_num_questions(self):
        """Test valid boundary values for num_questions."""
        options1 = ProcessingOptions(num_questions=1)
        assert options1.num_questions == 1
        options100 = ProcessingOptions(num_questions=100)
        assert options100.num_questions == 100

    def test_valid_boundary_summary_words(self):
        """Test valid boundary values for summary_max_words."""
        options50 = ProcessingOptions(summary_max_words=50)
        assert options50.summary_max_words == 50
        options2000 = ProcessingOptions(summary_max_words=2000)
        assert options2000.summary_max_words == 2000


class TestBulkFolderInfo:
    """Tests for BulkFolderInfo schema."""

    def test_from_dict(self):
        """Test creating from values."""
        info = BulkFolderInfo(
            folder_name="invoices",
            gcs_path="gs://bucket/org/bulk/invoices",
            document_count=5,
            total_size_bytes=1024000,
            org_id="test-org",
        )
        assert info.folder_name == "invoices"
        assert info.document_count == 5
        assert info.total_size_bytes == 1024000

    def test_default_values(self):
        """Test default values."""
        info = BulkFolderInfo(
            folder_name="test",
            gcs_path="gs://bucket/path",
            org_id="test-org",
        )
        assert info.document_count == 0
        assert info.total_size_bytes == 0
        assert info.created_at is None


class TestSignedUrlInfo:
    """Tests for SignedUrlInfo schema."""

    def test_create_signed_url_info(self):
        """Test creating signed URL info."""
        expires = datetime.utcnow()
        info = SignedUrlInfo(
            filename="doc.pdf",
            signed_url="https://signed-url.example.com",
            gcs_path="gs://bucket/path/doc.pdf",
            expires_at=expires,
            content_type="application/pdf",
        )
        assert info.filename == "doc.pdf"
        assert info.signed_url == "https://signed-url.example.com"
        assert info.expires_at == expires
        assert info.content_type == "application/pdf"

    def test_optional_content_type(self):
        """Test content_type is optional."""
        info = SignedUrlInfo(
            filename="doc.pdf",
            signed_url="https://signed-url.example.com",
            gcs_path="gs://bucket/path/doc.pdf",
            expires_at=datetime.utcnow(),
        )
        assert info.content_type is None


class TestDocumentItemInfo:
    """Tests for DocumentItemInfo schema."""

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "doc-123",
            "bulk_job_id": "job-456",
            "original_path": "gs://bucket/path/doc.pdf",
            "original_filename": "doc.pdf",
            "status": "pending",
        }
        info = DocumentItemInfo.from_dict(data)
        assert info.id == "doc-123"
        assert info.bulk_job_id == "job-456"
        assert info.status == DocumentItemStatus.PENDING

    def test_from_dict_with_optional_fields(self):
        """Test creating from dictionary with optional fields."""
        data = {
            "id": "doc-123",
            "bulk_job_id": "job-456",
            "original_path": "gs://bucket/path/doc.pdf",
            "original_filename": "doc.pdf",
            "parsed_path": "gs://bucket/parsed/doc.md",
            "status": "completed",
            "parse_time_ms": 1000,
            "index_time_ms": 500,
            "generation_time_ms": 2000,
            "total_time_ms": 3500,
            "token_usage": 1500,
            "llamaparse_pages": 3,
            "content_hash": "abc123",
        }
        info = DocumentItemInfo.from_dict(data)
        assert info.parsed_path == "gs://bucket/parsed/doc.md"
        assert info.status == DocumentItemStatus.COMPLETED
        assert info.parse_time_ms == 1000
        assert info.total_time_ms == 3500
        assert info.token_usage == 1500

    def test_default_values(self):
        """Test default values."""
        info = DocumentItemInfo(
            id="doc-123",
            bulk_job_id="job-456",
            original_path="gs://bucket/path/doc.pdf",
            original_filename="doc.pdf",
        )
        assert info.status == DocumentItemStatus.PENDING
        assert info.retry_count == 0
        assert info.token_usage == 0
        assert info.llamaparse_pages == 0


class TestBulkJobInfo:
    """Tests for BulkJobInfo schema."""

    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            "id": "job-123",
            "organization_id": "test-org",
            "folder_name": "invoices",
            "source_path": "gs://bucket/org/bulk/invoices",
            "total_documents": 10,
            "completed_count": 5,
            "failed_count": 1,
            "skipped_count": 0,
            "status": "processing",
        }
        info = BulkJobInfo.from_dict(data)
        assert info.id == "job-123"
        assert info.total_documents == 10
        assert info.completed_count == 5
        assert info.status == BulkJobStatus.PROCESSING

    def test_progress_percentage_calculation(self):
        """Test progress percentage property."""
        info = BulkJobInfo(
            id="job-123",
            organization_id="test-org",
            folder_name="test",
            source_path="gs://bucket/path",
            total_documents=10,
            completed_count=5,
            failed_count=2,
            skipped_count=1,
        )
        # (5 + 2 + 1) / 10 = 80%
        assert info.progress_percentage == 80.0

    def test_progress_percentage_zero_documents(self):
        """Test progress percentage with zero documents."""
        info = BulkJobInfo(
            id="job-123",
            organization_id="test-org",
            folder_name="test",
            source_path="gs://bucket/path",
            total_documents=0,
        )
        assert info.progress_percentage == 0.0

    def test_pending_count_calculation(self):
        """Test pending count property."""
        info = BulkJobInfo(
            id="job-123",
            organization_id="test-org",
            folder_name="test",
            source_path="gs://bucket/path",
            total_documents=10,
            completed_count=5,
            failed_count=2,
            skipped_count=1,
        )
        # 10 - (5 + 2 + 1) = 2 pending
        assert info.pending_count == 2

    def test_pending_count_no_negative(self):
        """Test pending count doesn't go negative."""
        info = BulkJobInfo(
            id="job-123",
            organization_id="test-org",
            folder_name="test",
            source_path="gs://bucket/path",
            total_documents=5,
            completed_count=10,  # Over-counted
        )
        assert info.pending_count == 0

    def test_default_values(self):
        """Test default values."""
        info = BulkJobInfo(
            id="job-123",
            organization_id="test-org",
            folder_name="test",
            source_path="gs://bucket/path",
        )
        assert info.total_documents == 0
        assert info.completed_count == 0
        assert info.failed_count == 0
        assert info.skipped_count == 0
        assert info.status == BulkJobStatus.PENDING
        assert info.documents is None

    def test_with_options(self):
        """Test BulkJobInfo with custom options."""
        data = {
            "id": "job-123",
            "organization_id": "test-org",
            "folder_name": "test",
            "source_path": "gs://bucket/path",
            "options": {
                "generate_summary": True,
                "generate_faqs": False,
                "num_faqs": 5,
            },
        }
        info = BulkJobInfo.from_dict(data)
        assert info.options.generate_summary is True
        assert info.options.generate_faqs is False
        assert info.options.num_faqs == 5


class TestBulkJobEvent:
    """Tests for BulkJobEvent schema."""

    def test_start_event(self):
        """Test creating start event."""
        event = BulkJobEvent(job_id="job-123", action="start")
        assert event.job_id == "job-123"
        assert event.action == "start"
        assert event.document_id is None

    def test_process_document_event(self):
        """Test creating process_document event."""
        event = BulkJobEvent(
            job_id="job-123",
            action="process_document",
            document_id="doc-456",
        )
        assert event.job_id == "job-123"
        assert event.action == "process_document"
        assert event.document_id == "doc-456"

    def test_complete_event(self):
        """Test creating complete event."""
        event = BulkJobEvent(job_id="job-123", action="complete")
        assert event.action == "complete"

    def test_cancel_event(self):
        """Test creating cancel event."""
        event = BulkJobEvent(job_id="job-123", action="cancel")
        assert event.action == "cancel"


class TestDocumentProcessingEvent:
    """Tests for DocumentProcessingEvent schema."""

    def test_create_event(self):
        """Test creating document processing event."""
        event = DocumentProcessingEvent(
            job_id="job-123",
            document_id="doc-456",
        )
        assert event.job_id == "job-123"
        assert event.document_id == "doc-456"
        assert event.action == "process"
        assert event.retry_count == 0

    def test_create_retry_event(self):
        """Test creating retry event."""
        event = DocumentProcessingEvent(
            job_id="job-123",
            document_id="doc-456",
            action="retry",
            retry_count=2,
        )
        assert event.action == "retry"
        assert event.retry_count == 2


class TestWebhookPayload:
    """Tests for WebhookPayload schema."""

    def test_org_id_extraction(self):
        """Test org_id property extraction from path."""
        payload = WebhookPayload(
            bucket="test-bucket",
            name="org123/bulk/folder/file.pdf",
        )
        assert payload.org_id == "org123"

    def test_folder_name_extraction(self):
        """Test folder_name property extraction from path."""
        payload = WebhookPayload(
            bucket="test-bucket",
            name="org123/bulk/invoices/file.pdf",
        )
        assert payload.folder_name == "invoices"

    def test_filename_extraction(self):
        """Test filename property extraction from path."""
        payload = WebhookPayload(
            bucket="test-bucket",
            name="org123/bulk/folder/document.pdf",
        )
        assert payload.filename == "document.pdf"

    def test_org_id_short_path(self):
        """Test org_id with short path."""
        payload = WebhookPayload(
            bucket="test-bucket",
            name="org123",
        )
        assert payload.org_id is None

    def test_folder_name_short_path(self):
        """Test folder_name with short path."""
        payload = WebhookPayload(
            bucket="test-bucket",
            name="org123/bulk",
        )
        assert payload.folder_name is None

    def test_default_values(self):
        """Test default values."""
        payload = WebhookPayload(
            bucket="test-bucket",
            name="org/bulk/folder/file.pdf",
        )
        assert payload.size == 0
        assert payload.content_type is None
        assert payload.time_created is None

    def test_with_all_fields(self):
        """Test with all fields populated."""
        payload = WebhookPayload(
            bucket="test-bucket",
            name="org/bulk/folder/file.pdf",
            size=1024,
            content_type="application/pdf",
            time_created="2024-01-01T00:00:00Z",
            updated="2024-01-01T00:00:00Z",
            metageneration="1",
        )
        assert payload.size == 1024
        assert payload.content_type == "application/pdf"
        assert payload.metageneration == "1"
