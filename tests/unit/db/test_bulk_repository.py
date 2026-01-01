"""Unit tests for bulk processing repository."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

# Mock the models before importing repository
mock_bulk_job = MagicMock()
mock_bulk_job_document = MagicMock()


class TestCreateBulkJob:
    """Tests for create_bulk_job function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_create_job_success(self, mock_db_session):
        """Test successful job creation."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.to_dict.return_value = {
            "id": "job-123",
            "organization_id": "test-org",
            "folder_name": "test-folder",
            "source_path": "gs://bucket/path",
            "status": "pending",
        }

        with patch("src.db.repositories.bulk_repository.db") as mock_db, \
             patch("src.db.repositories.bulk_repository.BulkJob") as MockJob:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session
            MockJob.return_value = mock_job

            result = await bulk_repository.create_bulk_job(
                organization_id="test-org",
                folder_name="test-folder",
                source_path="gs://bucket/path",
                total_documents=5,
            )

        assert result["id"] == "job-123"
        assert result["status"] == "pending"
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_job_with_options(self, mock_db_session):
        """Test job creation with processing options."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.id = "job-456"
        mock_job.to_dict.return_value = {
            "id": "job-456",
            "organization_id": "test-org",
            "folder_name": "test-folder",
            "source_path": "gs://bucket/path",
            "status": "pending",
            "options": {"generate_summary": True, "num_faqs": 5},
        }

        options = {"generate_summary": True, "num_faqs": 5}

        with patch("src.db.repositories.bulk_repository.db") as mock_db, \
             patch("src.db.repositories.bulk_repository.BulkJob") as MockJob:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session
            MockJob.return_value = mock_job

            result = await bulk_repository.create_bulk_job(
                organization_id="test-org",
                folder_name="test-folder",
                source_path="gs://bucket/path",
                options=options,
            )

        assert result["options"] == options


class TestGetBulkJob:
    """Tests for get_bulk_job function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_get_job_found(self, mock_db_session):
        """Test getting existing job."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.to_dict.return_value = {
            "id": "job-123",
            "organization_id": "test-org",
            "folder_name": "test-folder",
            "status": "processing",
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_bulk_job("job-123")

        assert result is not None
        assert result["id"] == "job-123"
        assert result["status"] == "processing"

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, mock_db_session):
        """Test getting non-existent job."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_bulk_job("nonexistent")

        assert result is None


class TestGetBulkJobByFolder:
    """Tests for get_bulk_job_by_folder function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_job_by_folder_found(self, mock_db_session):
        """Test finding job by folder name."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.to_dict.return_value = {
            "id": "job-123",
            "organization_id": "test-org",
            "folder_name": "invoices",
            "status": "pending",
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_bulk_job_by_folder(
                organization_id="test-org",
                folder_name="invoices",
            )

        assert result is not None
        assert result["folder_name"] == "invoices"

    @pytest.mark.asyncio
    async def test_get_job_by_folder_with_status(self, mock_db_session):
        """Test finding job by folder with status filter."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.to_dict.return_value = {
            "id": "job-123",
            "folder_name": "invoices",
            "status": "processing",
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_bulk_job_by_folder(
                organization_id="test-org",
                folder_name="invoices",
                status="processing",
            )

        assert result["status"] == "processing"


class TestListBulkJobs:
    """Tests for list_bulk_jobs function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_list_jobs_success(self, mock_db_session):
        """Test listing jobs for organization."""
        from src.db.repositories import bulk_repository

        mock_jobs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "job-1", "status": "pending"})),
            MagicMock(to_dict=MagicMock(return_value={"id": "job-2", "status": "completed"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.list_bulk_jobs(
                organization_id="test-org",
            )

        assert len(result) == 2
        assert result[0]["id"] == "job-1"

    @pytest.mark.asyncio
    async def test_list_jobs_with_status_filter(self, mock_db_session):
        """Test listing jobs with status filter."""
        from src.db.repositories import bulk_repository

        mock_jobs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "job-1", "status": "processing"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.list_bulk_jobs(
                organization_id="test-org",
                status="processing",
            )

        assert len(result) == 1
        assert result[0]["status"] == "processing"

    @pytest.mark.asyncio
    async def test_list_jobs_with_pagination(self, mock_db_session):
        """Test listing jobs with pagination."""
        from src.db.repositories import bulk_repository

        mock_jobs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "job-2"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.list_bulk_jobs(
                organization_id="test-org",
                limit=1,
                offset=1,
            )

        assert len(result) == 1


class TestCountBulkJobs:
    """Tests for count_bulk_jobs function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_count_jobs_success(self, mock_db_session):
        """Test counting jobs."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar.return_value = 5
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.count_bulk_jobs(organization_id="test-org")

        assert result == 5

    @pytest.mark.asyncio
    async def test_count_jobs_with_status(self, mock_db_session):
        """Test counting jobs with status filter."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar.return_value = 2
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.count_bulk_jobs(
                organization_id="test-org",
                status="processing",
            )

        assert result == 2


class TestUpdateBulkJobStatus:
    """Tests for update_bulk_job_status function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_update_status_success(self, mock_db_session):
        """Test updating job status."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.status = "pending"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.update_bulk_job_status(
                job_id="job-123",
                status="processing",
            )

        assert result is True
        assert mock_job.status == "processing"

    @pytest.mark.asyncio
    async def test_update_status_job_not_found(self, mock_db_session):
        """Test updating non-existent job."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.update_bulk_job_status(
                job_id="nonexistent",
                status="processing",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_status_with_timestamps(self, mock_db_session):
        """Test updating job status with timestamps."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.started_at = None
        mock_job.completed_at = None

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        start_time = datetime.utcnow()

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.update_bulk_job_status(
                job_id="job-123",
                status="processing",
                started_at=start_time,
            )

        assert result is True
        assert mock_job.started_at == start_time


class TestIncrementJobCounts:
    """Tests for job count increment functions."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_increment_completed(self, mock_db_session):
        """Test incrementing completed count."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.completed_count = 0
        mock_job.total_tokens_used = 0
        mock_job.total_llamaparse_pages = 0

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.increment_job_completed(
                job_id="job-123",
                token_usage=100,
                llamaparse_pages=2,
            )

        assert result is True
        assert mock_job.completed_count == 1
        assert mock_job.total_tokens_used == 100
        assert mock_job.total_llamaparse_pages == 2

    @pytest.mark.asyncio
    async def test_increment_failed(self, mock_db_session):
        """Test incrementing failed count."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.failed_count = 1

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.increment_job_failed(job_id="job-123")

        assert result is True
        assert mock_job.failed_count == 2

    @pytest.mark.asyncio
    async def test_increment_skipped(self, mock_db_session):
        """Test incrementing skipped count."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.skipped_count = 0

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.increment_job_skipped(job_id="job-123")

        assert result is True
        assert mock_job.skipped_count == 1


class TestCreateDocumentItem:
    """Tests for create_document_item function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.flush = AsyncMock()
        session.add = MagicMock()
        return session

    @pytest.mark.asyncio
    async def test_create_document_success(self, mock_db_session):
        """Test creating document item."""
        from src.db.repositories import bulk_repository

        mock_doc = MagicMock()
        mock_doc.id = "doc-123"
        mock_doc.to_dict.return_value = {
            "id": "doc-123",
            "bulk_job_id": "job-456",
            "original_path": "gs://bucket/path/doc.pdf",
            "original_filename": "doc.pdf",
            "status": "pending",
        }

        with patch("src.db.repositories.bulk_repository.db") as mock_db, \
             patch("src.db.repositories.bulk_repository.BulkJobDocument") as MockDoc:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session
            MockDoc.return_value = mock_doc

            result = await bulk_repository.create_document_item(
                bulk_job_id="job-456",
                original_path="gs://bucket/path/doc.pdf",
                original_filename="doc.pdf",
            )

        assert result["id"] == "doc-123"
        assert result["status"] == "pending"
        mock_db_session.add.assert_called_once()


class TestGetDocumentItem:
    """Tests for get_document_item function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_document_found(self, mock_db_session):
        """Test getting existing document."""
        from src.db.repositories import bulk_repository

        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "id": "doc-123",
            "status": "completed",
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_document_item("doc-123")

        assert result is not None
        assert result["id"] == "doc-123"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mock_db_session):
        """Test getting non-existent document."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_document_item("nonexistent")

        assert result is None


class TestGetDocumentByPath:
    """Tests for get_document_by_path function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_by_path_found(self, mock_db_session):
        """Test finding document by path."""
        from src.db.repositories import bulk_repository

        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "id": "doc-123",
            "original_path": "gs://bucket/path/doc.pdf",
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_document_by_path(
                bulk_job_id="job-456",
                original_path="gs://bucket/path/doc.pdf",
            )

        assert result is not None
        assert result["original_path"] == "gs://bucket/path/doc.pdf"


class TestGetAllDocumentItems:
    """Tests for get_all_document_items function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_all_success(self, mock_db_session):
        """Test getting all documents in job."""
        from src.db.repositories import bulk_repository

        mock_docs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "doc-1", "status": "pending"})),
            MagicMock(to_dict=MagicMock(return_value={"id": "doc-2", "status": "completed"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_docs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_all_document_items("job-123")

        assert len(result) == 2


class TestGetPendingDocuments:
    """Tests for get_pending_documents function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_pending_success(self, mock_db_session):
        """Test getting pending documents."""
        from src.db.repositories import bulk_repository

        mock_docs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "doc-1", "status": "pending"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_docs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_pending_documents("job-123")

        assert len(result) == 1
        assert result[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_pending_with_limit(self, mock_db_session):
        """Test getting pending documents with limit."""
        from src.db.repositories import bulk_repository

        mock_docs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "doc-1"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_docs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_pending_documents("job-123", limit=5)

        assert len(result) == 1


class TestGetFailedDocuments:
    """Tests for get_failed_documents function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_failed_success(self, mock_db_session):
        """Test getting failed documents."""
        from src.db.repositories import bulk_repository

        mock_docs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "doc-1", "status": "failed", "error_message": "Parse error"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_docs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_failed_documents("job-123")

        assert len(result) == 1
        assert result[0]["status"] == "failed"


class TestCountInProgressDocuments:
    """Tests for count_in_progress_documents function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_count_in_progress(self, mock_db_session):
        """Test counting in-progress documents."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar.return_value = 3
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.count_in_progress_documents("job-123")

        assert result == 3


class TestUpdateDocumentItem:
    """Tests for update_document_item function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_update_document_success(self, mock_db_session):
        """Test updating document item."""
        from src.db.repositories import bulk_repository

        mock_doc = MagicMock()
        mock_doc.status = "pending"

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.update_document_item(
                document_id="doc-123",
                status="parsing",
            )

        assert result is True
        assert mock_doc.status == "parsing"

    @pytest.mark.asyncio
    async def test_update_document_not_found(self, mock_db_session):
        """Test updating non-existent document."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.update_document_item(
                document_id="nonexistent",
                status="completed",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_document_all_fields(self, mock_db_session):
        """Test updating all document fields."""
        from src.db.repositories import bulk_repository

        mock_doc = MagicMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.update_document_item(
                document_id="doc-123",
                status="completed",
                parsed_path="gs://bucket/parsed/doc.md",
                parse_time_ms=1000,
                index_time_ms=500,
                generation_time_ms=2000,
                total_time_ms=3500,
                token_usage=1500,
                llamaparse_pages=3,
                content_hash="abc123",
            )

        assert result is True
        assert mock_doc.parsed_path == "gs://bucket/parsed/doc.md"
        assert mock_doc.token_usage == 1500


class TestResetDocumentForRetry:
    """Tests for reset_document_for_retry function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_reset_for_retry(self, mock_db_session):
        """Test resetting document for retry."""
        from src.db.repositories import bulk_repository

        mock_doc = MagicMock()
        mock_doc.status = "failed"
        mock_doc.retry_count = 1

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.reset_document_for_retry("doc-123")

        assert result is True
        assert mock_doc.status == "pending"
        assert mock_doc.error_message is None
        assert mock_doc.retry_count == 2


class TestBulkResetFailedDocuments:
    """Tests for bulk_reset_failed_documents function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_bulk_reset_success(self, mock_db_session):
        """Test bulk resetting failed documents."""
        from src.db.repositories import bulk_repository

        # Mock the update result
        mock_update_result = MagicMock()
        mock_update_result.rowcount = 3

        # Mock the job query
        mock_job = MagicMock()
        mock_job.failed_count = 3

        mock_job_result = MagicMock()
        mock_job_result.scalar_one_or_none.return_value = mock_job

        mock_db_session.execute.side_effect = [mock_update_result, mock_job_result]

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.bulk_reset_failed_documents("job-123")

        assert result == 3
        assert mock_job.failed_count == 0
        assert mock_job.status == "processing"


class TestDeleteBulkJob:
    """Tests for delete_bulk_job function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.delete = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_delete_job_success(self, mock_db_session):
        """Test deleting job."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.delete_bulk_job("job-123")

        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_job)

    @pytest.mark.asyncio
    async def test_delete_job_not_found(self, mock_db_session):
        """Test deleting non-existent job."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.delete_bulk_job("nonexistent")

        assert result is False


class TestGetActiveJobsForFolder:
    """Tests for get_active_jobs_for_folder function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_active_jobs(self, mock_db_session):
        """Test getting active jobs for folder."""
        from src.db.repositories import bulk_repository

        mock_jobs = [
            MagicMock(to_dict=MagicMock(return_value={"id": "job-1", "status": "pending"})),
            MagicMock(to_dict=MagicMock(return_value={"id": "job-2", "status": "processing"})),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = mock_jobs
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_active_jobs_for_folder(
                organization_id="test-org",
                folder_name="invoices",
            )

        assert len(result) == 2


class TestFindActiveJobForFolder:
    """Tests for find_active_job_for_folder function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_find_active_job_found(self, mock_db_session):
        """Test finding active job for folder."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.to_dict.return_value = {
            "id": "job-123",
            "status": "pending",
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.find_active_job_for_folder(
                organization_id="test-org",
                folder_name="invoices",
            )

        assert result is not None
        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_find_active_job_not_found(self, mock_db_session):
        """Test no active job for folder."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.find_active_job_for_folder(
                organization_id="test-org",
                folder_name="invoices",
            )

        assert result is None


class TestCountDocumentsInJob:
    """Tests for count_documents_in_job function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_count_documents(self, mock_db_session):
        """Test counting documents in job."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar.return_value = 10
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.count_documents_in_job("job-123")

        assert result == 10


class TestIncrementTotalDocuments:
    """Tests for increment_total_documents function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_increment_total(self, mock_db_session):
        """Test incrementing total document count."""
        from src.db.repositories import bulk_repository

        mock_job = MagicMock()
        mock_job.total_documents = 5

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_job
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.increment_total_documents("job-123")

        assert result is True
        assert mock_job.total_documents == 6


class TestGetLatestDocumentInJob:
    """Tests for get_latest_document_in_job function."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_get_latest_found(self, mock_db_session):
        """Test getting latest document."""
        from src.db.repositories import bulk_repository

        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "id": "doc-latest",
            "original_filename": "latest.pdf",
        }

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_doc
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_latest_document_in_job("job-123")

        assert result is not None
        assert result["id"] == "doc-latest"

    @pytest.mark.asyncio
    async def test_get_latest_not_found(self, mock_db_session):
        """Test getting latest when no documents."""
        from src.db.repositories import bulk_repository

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("src.db.repositories.bulk_repository.db") as mock_db:
            mock_db.session.return_value.__aenter__.return_value = mock_db_session

            result = await bulk_repository.get_latest_document_in_job("job-123")

        assert result is None
