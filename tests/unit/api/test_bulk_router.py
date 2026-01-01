"""Unit tests for bulk processing API router."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
from datetime import datetime

TEST_ORG_ID = "test-org-123"


class TestCreateBulkFolderEndpoint:
    """Tests for POST /folders endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_create_folder_success(self, client, headers, mock_folder_manager):
        """Test successful folder creation."""
        mock_folder_manager.folder_exists = AsyncMock(return_value=False)

        with patch("src.api.routers.bulk.get_folder_manager", return_value=mock_folder_manager):
            response = client.post(
                "/api/v1/bulk/folders",
                json={"folder_name": "test-folder"},
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["folder"]["folder_name"] == "test-folder"

    def test_create_folder_already_exists(self, client, headers, mock_folder_manager):
        """Test error when folder already exists."""
        mock_folder_manager.folder_exists = AsyncMock(return_value=True)

        with patch("src.api.routers.bulk.get_folder_manager", return_value=mock_folder_manager):
            response = client.post(
                "/api/v1/bulk/folders",
                json={"folder_name": "existing-folder"},
                headers=headers,
            )

        assert response.status_code == 400


class TestListBulkFoldersEndpoint:
    """Tests for GET /folders endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_list_folders_success(self, client, headers, mock_folder_manager):
        """Test listing folders for organization."""
        from src.bulk.schemas import BulkFolderInfo

        mock_folder_manager.list_folders = AsyncMock(return_value=[
            BulkFolderInfo(
                folder_name="folder1",
                gcs_path="gs://bucket/org/bulk/folder1",
                document_count=5,
                total_size_bytes=1024,
                org_id=TEST_ORG_ID,
            ),
            BulkFolderInfo(
                folder_name="folder2",
                gcs_path="gs://bucket/org/bulk/folder2",
                document_count=3,
                total_size_bytes=512,
                org_id=TEST_ORG_ID,
            ),
        ])

        with patch("src.api.routers.bulk.get_folder_manager", return_value=mock_folder_manager):
            response = client.get(
                "/api/v1/bulk/folders",
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 2
        assert len(data["folders"]) == 2

    def test_list_folders_empty(self, client, headers, mock_folder_manager):
        """Test empty folder list response."""
        mock_folder_manager.list_folders = AsyncMock(return_value=[])

        with patch("src.api.routers.bulk.get_folder_manager", return_value=mock_folder_manager):
            response = client.get(
                "/api/v1/bulk/folders",
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert data["folders"] == []


class TestGenerateUploadUrlsEndpoint:
    """Tests for POST /folders/{name}/upload-urls endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_generate_urls_success(self, client, headers, mock_folder_manager, mock_bulk_config):
        """Test successful signed URL generation."""
        from src.bulk.schemas import SignedUrlInfo

        mock_folder_manager.generate_upload_urls = AsyncMock(return_value=[
            SignedUrlInfo(
                filename="doc1.pdf",
                signed_url="https://signed-url-1.example.com",
                gcs_path="gs://bucket/org/bulk/folder/doc1.pdf",
                expires_at=datetime.utcnow(),
            ),
            SignedUrlInfo(
                filename="doc2.pdf",
                signed_url="https://signed-url-2.example.com",
                gcs_path="gs://bucket/org/bulk/folder/doc2.pdf",
                expires_at=datetime.utcnow(),
            ),
        ])

        with patch("src.api.routers.bulk.get_folder_manager", return_value=mock_folder_manager), \
             patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            response = client.post(
                "/api/v1/bulk/folders/test-folder/upload-urls",
                json={"filenames": ["doc1.pdf", "doc2.pdf"]},
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 2
        assert len(data["urls"]) == 2

    def test_generate_urls_too_many_files(self, client, headers, mock_folder_manager, mock_bulk_config):
        """Test error when exceeding max files limit."""
        mock_bulk_config.max_documents_per_folder = 5
        filenames = [f"doc{i}.pdf" for i in range(10)]

        with patch("src.api.routers.bulk.get_folder_manager", return_value=mock_folder_manager), \
             patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            response = client.post(
                "/api/v1/bulk/folders/test-folder/upload-urls",
                json={"filenames": filenames},
                headers=headers,
            )

        assert response.status_code == 400


class TestListFolderDocumentsEndpoint:
    """Tests for GET /folders/{name}/documents endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_list_documents_success(self, client, headers, mock_folder_manager):
        """Test listing documents in folder."""
        mock_folder_manager.list_documents = AsyncMock(return_value=[
            "gs://bucket/org/bulk/folder/doc1.pdf",
            "gs://bucket/org/bulk/folder/doc2.pdf",
        ])

        with patch("src.api.routers.bulk.get_folder_manager", return_value=mock_folder_manager):
            response = client.get(
                "/api/v1/bulk/folders/test-folder/documents",
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 2
        assert len(data["documents"]) == 2


class TestSubmitBulkJobEndpoint:
    """Tests for POST /jobs endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_submit_job_success(self, client, headers, mock_bulk_service, mock_bulk_queue):
        """Test successful job submission."""
        with patch("src.api.routers.bulk.get_bulk_service", return_value=mock_bulk_service), \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):
            response = client.post(
                "/api/v1/bulk/jobs",
                json={
                    "folder_name": "test-folder",
                    "generate_summary": True,
                    "generate_faqs": True,
                    "generate_questions": False,
                },
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        mock_bulk_queue.enqueue.assert_called_once()

    def test_submit_job_folder_not_found(self, client, headers, mock_bulk_service, mock_bulk_queue):
        """Test error when folder doesn't exist."""
        mock_bulk_service.create_job = AsyncMock(side_effect=ValueError("Folder not found"))

        with patch("src.api.routers.bulk.get_bulk_service", return_value=mock_bulk_service), \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):
            response = client.post(
                "/api/v1/bulk/jobs",
                json={"folder_name": "nonexistent"},
                headers=headers,
            )

        assert response.status_code == 400


class TestGetBulkJobStatusEndpoint:
    """Tests for GET /jobs/{job_id} endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_get_job_status_success(self, client, headers):
        """Test getting job status."""
        from src.bulk.schemas import BulkJobInfo, BulkJobStatus, ProcessingOptions

        mock_job = BulkJobInfo(
            id="job-123",
            organization_id=TEST_ORG_ID,
            folder_name="test-folder",
            source_path="gs://bucket/org/bulk/test-folder",
            total_documents=5,
            completed_count=3,
            status=BulkJobStatus.PROCESSING,
            options=ProcessingOptions(),
        )

        mock_service = MagicMock()
        mock_service.get_job_status = AsyncMock(return_value=mock_job)

        with patch("src.api.routers.bulk.get_bulk_service", return_value=mock_service):
            response = client.get(
                "/api/v1/bulk/jobs/job-123",
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "job" in data
        assert "progress_percentage" in data

    def test_get_job_status_not_found(self, client, headers):
        """Test 404 when job not found."""
        mock_service = MagicMock()
        mock_service.get_job_status = AsyncMock(return_value=None)

        with patch("src.api.routers.bulk.get_bulk_service", return_value=mock_service):
            response = client.get(
                "/api/v1/bulk/jobs/nonexistent",
                headers=headers,
            )

        assert response.status_code == 404

    def test_get_job_status_wrong_org(self, client, headers):
        """Test 404 when job belongs to different org."""
        from src.bulk.schemas import BulkJobInfo, BulkJobStatus, ProcessingOptions

        wrong_org_job = BulkJobInfo(
            id="job-123",
            organization_id="other-org",
            folder_name="test-folder",
            source_path="gs://bucket/other-org/bulk/test-folder",
            status=BulkJobStatus.PENDING,
            options=ProcessingOptions(),
        )

        mock_service = MagicMock()
        mock_service.get_job_status = AsyncMock(return_value=wrong_org_job)

        with patch("src.api.routers.bulk.get_bulk_service", return_value=mock_service):
            response = client.get(
                "/api/v1/bulk/jobs/job-123",
                headers=headers,
            )

        assert response.status_code == 404


class TestListBulkJobsEndpoint:
    """Tests for GET /jobs endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_list_jobs_success(self, client, headers, sample_bulk_job_dict):
        """Test listing jobs for organization."""
        sample_bulk_job_dict["organization_id"] = TEST_ORG_ID

        with patch("src.db.repositories.bulk_repository.list_bulk_jobs", new_callable=AsyncMock) as mock_list, \
             patch("src.db.repositories.bulk_repository.count_bulk_jobs", new_callable=AsyncMock) as mock_count:

            mock_list.return_value = [sample_bulk_job_dict]
            mock_count.return_value = 1

            response = client.get(
                "/api/v1/bulk/jobs",
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total"] == 1


class TestCancelBulkJobEndpoint:
    """Tests for POST /jobs/{job_id}/cancel endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_cancel_job_success(self, client, headers, sample_bulk_job_dict, mock_bulk_queue):
        """Test successful job cancellation."""
        sample_bulk_job_dict["organization_id"] = TEST_ORG_ID
        sample_bulk_job_dict["status"] = "processing"

        with patch("src.db.repositories.bulk_repository.get_bulk_job", new_callable=AsyncMock) as mock_get, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_get.return_value = sample_bulk_job_dict

            response = client.post(
                "/api/v1/bulk/jobs/job-123/cancel",
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_bulk_queue.enqueue.assert_called_once()

    def test_cancel_job_already_completed(self, client, headers, sample_bulk_job_dict):
        """Test error when job already completed."""
        sample_bulk_job_dict["organization_id"] = TEST_ORG_ID
        sample_bulk_job_dict["status"] = "completed"

        with patch("src.db.repositories.bulk_repository.get_bulk_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = sample_bulk_job_dict

            response = client.post(
                "/api/v1/bulk/jobs/job-123/cancel",
                headers=headers,
            )

        assert response.status_code == 400

    def test_cancel_job_not_found(self, client, headers):
        """Test 404 when job not found."""
        with patch("src.db.repositories.bulk_repository.get_bulk_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.post(
                "/api/v1/bulk/jobs/nonexistent/cancel",
                headers=headers,
            )

        assert response.status_code == 404


class TestRetryFailedDocumentsEndpoint:
    """Tests for POST /jobs/{job_id}/retry endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    def test_retry_all_failed_success(self, client, headers, sample_bulk_job_dict, mock_bulk_service):
        """Test retrying all failed documents."""
        sample_bulk_job_dict["organization_id"] = TEST_ORG_ID
        mock_bulk_service.retry_failed_documents = AsyncMock(return_value=3)

        with patch("src.db.repositories.bulk_repository.get_bulk_job", new_callable=AsyncMock) as mock_get, \
             patch("src.api.routers.bulk.get_bulk_service", return_value=mock_bulk_service):

            mock_get.return_value = sample_bulk_job_dict

            response = client.post(
                "/api/v1/bulk/jobs/job-123/retry",
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["retried_count"] == 3

    def test_retry_job_not_found(self, client, headers):
        """Test 404 when job not found."""
        with patch("src.db.repositories.bulk_repository.get_bulk_job", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            response = client.post(
                "/api/v1/bulk/jobs/nonexistent/retry",
                headers=headers,
            )

        assert response.status_code == 404


class TestWebhookEndpoint:
    """Tests for POST /webhook/document-uploaded endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    def test_webhook_success(self, client, mock_bulk_config):
        """Test successful webhook processing."""
        mock_bulk_config.webhook_enabled = True
        mock_bulk_config.webhook_secret = None

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.bulk.webhook_handler.handle_document_uploaded", new_callable=AsyncMock) as mock_handler:

            mock_handler.return_value = {
                "success": True,
                "message": "Document added",
                "job_id": "job-123",
                "document_id": "doc-456",
            }

            response = client.post(
                "/api/v1/bulk/webhook/document-uploaded",
                json={
                    "bucket": "test-bucket",
                    "name": "org123/bulk/folder/doc.pdf",
                    "size": 1024,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == "job-123"
        assert data["document_id"] == "doc-456"

    def test_webhook_disabled(self, client, mock_bulk_config):
        """Test 403 when webhook disabled."""
        mock_bulk_config.webhook_enabled = False

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            response = client.post(
                "/api/v1/bulk/webhook/document-uploaded",
                json={
                    "bucket": "test-bucket",
                    "name": "org123/bulk/folder/doc.pdf",
                },
            )

        assert response.status_code == 403

    def test_webhook_invalid_secret(self, client, mock_bulk_config):
        """Test 401 when webhook secret invalid."""
        mock_bulk_config.webhook_enabled = True
        mock_bulk_config.webhook_secret = "correct-secret"

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            response = client.post(
                "/api/v1/bulk/webhook/document-uploaded",
                json={
                    "bucket": "test-bucket",
                    "name": "org123/bulk/folder/doc.pdf",
                },
                headers={"X-Webhook-Secret": "wrong-secret"},
            )

        assert response.status_code == 401

    def test_webhook_valid_secret(self, client, mock_bulk_config):
        """Test success with valid webhook secret."""
        mock_bulk_config.webhook_enabled = True
        mock_bulk_config.webhook_secret = "correct-secret"

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.bulk.webhook_handler.handle_document_uploaded", new_callable=AsyncMock) as mock_handler:

            mock_handler.return_value = {
                "success": True,
                "message": "Document added",
            }

            response = client.post(
                "/api/v1/bulk/webhook/document-uploaded",
                json={
                    "bucket": "test-bucket",
                    "name": "org123/bulk/folder/doc.pdf",
                },
                headers={"X-Webhook-Secret": "correct-secret"},
            )

        assert response.status_code == 200


# =============================================================================
# UPLOAD ENDPOINT TESTS (New direct upload feature)
# =============================================================================


class TestUploadBulkFilesEndpoint:
    """Tests for POST /upload endpoint (direct bulk upload)."""

    @pytest.fixture
    def client(self):
        """Create test client for FastAPI app."""
        from src.api.app import create_app
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def headers(self):
        """Default headers with organization ID."""
        return {"X-Organization-ID": TEST_ORG_ID}

    @pytest.fixture
    def mock_storage(self):
        """Mock GCS storage for upload tests."""
        storage = MagicMock()
        storage.bucket_name = "test-bucket"
        storage.save_bytes = AsyncMock(
            return_value="gs://test-bucket/test-org/original/test-folder/test.pdf"
        )
        return storage

    def test_upload_single_file_success(
        self, client, headers, mock_bulk_config, mock_storage, mock_bulk_queue
    ):
        """Test successful single file upload."""
        # Create test file content
        file_content = b"%PDF-1.4 test pdf content"

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage), \
             patch("src.api.routers.bulk.bulk_repository") as mock_repo, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.create_document_item = AsyncMock(return_value={"id": "doc-456"})
            mock_repo.update_job_document_count = AsyncMock(return_value=True)

            response = client.post(
                "/api/v1/bulk/upload",
                data={
                    "folder_name": "test-folder",
                    "generate_summary": "true",
                    "generate_faqs": "true",
                    "generate_questions": "true",
                    "auto_start": "true",
                },
                files=[("files", ("test.pdf", file_content, "application/pdf"))],
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["job_id"] == "job-123"
        assert data["folder_name"] == "test-folder"
        assert data["total_documents"] == 1
        assert len(data["uploaded_files"]) == 1
        assert len(data["failed_files"]) == 0
        assert data["uploaded_files"][0]["document_id"] == "doc-456"
        mock_bulk_queue.enqueue.assert_called_once()

    def test_upload_multiple_files_success(
        self, client, headers, mock_bulk_config, mock_storage, mock_bulk_queue
    ):
        """Test successful multiple file upload."""
        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage), \
             patch("src.api.routers.bulk.bulk_repository") as mock_repo, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.create_document_item = AsyncMock(side_effect=[
                {"id": "doc-1"},
                {"id": "doc-2"},
                {"id": "doc-3"},
            ])
            mock_repo.update_job_document_count = AsyncMock(return_value=True)

            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=[
                    ("files", ("doc1.pdf", b"pdf content 1", "application/pdf")),
                    ("files", ("doc2.docx", b"docx content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
                    ("files", ("doc3.txt", b"text content", "text/plain")),
                ],
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_documents"] == 3
        assert len(data["uploaded_files"]) == 3

    def test_upload_exceeds_file_limit(self, client, headers, mock_bulk_config):
        """Test error when exceeding max file limit."""
        mock_bulk_config.max_documents_per_folder = 3

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            # Create 5 files (more than limit of 3)
            files = [
                ("files", (f"doc{i}.pdf", b"content", "application/pdf"))
                for i in range(5)
            ]

            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=files,
                headers=headers,
            )

        assert response.status_code == 400
        assert "Maximum is 3" in response.json()["detail"]

    def test_upload_unsupported_file_type(
        self, client, headers, mock_bulk_config, mock_storage, mock_bulk_queue
    ):
        """Test handling of unsupported file types."""
        mock_bulk_config.supported_extensions = [".pdf", ".docx", ".txt"]

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage), \
             patch("src.api.routers.bulk.bulk_repository") as mock_repo, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.create_document_item = AsyncMock(return_value={"id": "doc-1"})
            mock_repo.update_job_document_count = AsyncMock(return_value=True)

            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=[
                    ("files", ("valid.pdf", b"pdf content", "application/pdf")),
                    ("files", ("invalid.exe", b"exe content", "application/x-executable")),
                ],
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_documents"] == 1
        assert len(data["uploaded_files"]) == 1
        assert len(data["failed_files"]) == 1
        assert "Unsupported file type" in data["failed_files"][0]["error"]

    def test_upload_file_too_large(
        self, client, headers, mock_bulk_config, mock_storage, mock_bulk_queue
    ):
        """Test handling of oversized files."""
        mock_bulk_config.max_file_size_mb = 1  # 1MB limit

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage), \
             patch("src.api.routers.bulk.bulk_repository") as mock_repo, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.create_document_item = AsyncMock(return_value={"id": "doc-1"})
            mock_repo.update_job_document_count = AsyncMock(return_value=True)

            # Create a file larger than 1MB
            large_content = b"x" * (2 * 1024 * 1024)  # 2MB

            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=[
                    ("files", ("small.pdf", b"small content", "application/pdf")),
                    ("files", ("large.pdf", large_content, "application/pdf")),
                ],
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["uploaded_files"]) == 1
        assert len(data["failed_files"]) == 1
        assert "exceeds" in data["failed_files"][0]["error"]

    def test_upload_with_auto_start_false(
        self, client, headers, mock_bulk_config, mock_storage, mock_bulk_queue
    ):
        """Test upload without auto-starting processing."""
        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage), \
             patch("src.api.routers.bulk.bulk_repository") as mock_repo, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.create_document_item = AsyncMock(return_value={"id": "doc-1"})
            mock_repo.update_job_document_count = AsyncMock(return_value=True)

            response = client.post(
                "/api/v1/bulk/upload",
                data={
                    "folder_name": "test-folder",
                    "auto_start": "false",
                },
                files=[("files", ("test.pdf", b"content", "application/pdf"))],
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "pending"
        assert "Ready to process" in data["message"]
        mock_bulk_queue.enqueue.assert_not_called()

    def test_upload_no_files_provided(self, client, headers, mock_bulk_config):
        """Test error when no files provided."""
        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=[],
                headers=headers,
            )

        assert response.status_code == 422  # Validation error

    def test_upload_missing_folder_name(self, client, headers, mock_bulk_config):
        """Test error when folder name missing."""
        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            response = client.post(
                "/api/v1/bulk/upload",
                data={},
                files=[("files", ("test.pdf", b"content", "application/pdf"))],
                headers=headers,
            )

        assert response.status_code == 422  # Validation error

    def test_upload_empty_folder_name(
        self, client, headers, mock_bulk_config, mock_storage
    ):
        """Test error when folder name is empty."""
        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage):

            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "   "},
                files=[("files", ("test.pdf", b"content", "application/pdf"))],
                headers=headers,
            )

        assert response.status_code == 400
        assert "Folder name is required" in response.json()["detail"]

    def test_upload_missing_org_header(self, client, mock_bulk_config):
        """Test error when organization header missing."""
        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config):
            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=[("files", ("test.pdf", b"content", "application/pdf"))],
            )

        # Should fail with 400 or 422 due to missing org header
        assert response.status_code in [400, 422]

    def test_upload_all_files_fail(
        self, client, headers, mock_bulk_config, mock_storage, mock_bulk_queue
    ):
        """Test response when all files fail to upload."""
        mock_bulk_config.supported_extensions = [".pdf"]

        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage), \
             patch("src.api.routers.bulk.bulk_repository") as mock_repo, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.update_job_document_count = AsyncMock(return_value=True)

            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=[
                    ("files", ("bad1.exe", b"content", "application/x-executable")),
                    ("files", ("bad2.bat", b"content", "application/x-bat")),
                ],
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["total_documents"] == 0
        assert len(data["failed_files"]) == 2
        assert "All" in data["message"] and "failed" in data["message"]
        mock_bulk_queue.enqueue.assert_not_called()

    def test_upload_sanitizes_filename(
        self, client, headers, mock_bulk_config, mock_storage, mock_bulk_queue
    ):
        """Test that filenames are sanitized properly."""
        with patch("src.api.routers.bulk.get_bulk_config", return_value=mock_bulk_config), \
             patch("src.api.routers.bulk.get_storage", return_value=mock_storage), \
             patch("src.api.routers.bulk.bulk_repository") as mock_repo, \
             patch("src.api.routers.bulk.get_bulk_queue", return_value=mock_bulk_queue):

            mock_repo.create_bulk_job = AsyncMock(return_value={"id": "job-123"})
            mock_repo.create_document_item = AsyncMock(return_value={"id": "doc-1"})
            mock_repo.update_job_document_count = AsyncMock(return_value=True)

            response = client.post(
                "/api/v1/bulk/upload",
                data={"folder_name": "test-folder"},
                files=[
                    ("files", ("My Document (1).pdf", b"content", "application/pdf")),
                ],
                headers=headers,
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Check that filename was sanitized (spaces to underscores, special chars removed)
        sanitized_name = data["uploaded_files"][0]["filename"]
        assert " " not in sanitized_name
        assert "(" not in sanitized_name
        assert ")" not in sanitized_name
