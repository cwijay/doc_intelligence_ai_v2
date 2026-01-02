"""
Bulk processing service.

Orchestrates bulk document processing jobs:
- Job creation and validation
- Document processing coordination
- Progress tracking and finalization
- Quota enforcement
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from src.core.patterns import ThreadSafeSingleton
from src.core.executors import get_executors
from src.db.repositories import bulk_repository
from src.utils.gcs_utils import extract_gcs_path_parts

from .config import get_bulk_config, BulkProcessingConfig
from .schemas import (
    BulkJobStatus,
    DocumentItemStatus,
    ProcessingOptions,
    BulkJobInfo,
    DocumentItemInfo,
    BulkJobEvent,
)
from .folder_manager import get_folder_manager
from .state_graph import (
    get_document_graph,
    create_initial_state,
)

logger = logging.getLogger(__name__)


# =============================================================================
# JOB STATUS CACHE
# =============================================================================

# Cache for job status to reduce DB polling load
# Format: {job_id: (BulkJobInfo, timestamp)}
_job_status_cache: Dict[str, Tuple[BulkJobInfo, datetime]] = {}
_job_cache_lock = threading.Lock()

# Cache TTL in seconds (short TTL since job status changes frequently during processing)
JOB_STATUS_CACHE_TTL_SECONDS = 3


def _get_cached_job_status(job_id: str) -> Optional[BulkJobInfo]:
    """Get job status from cache if not expired."""
    with _job_cache_lock:
        if job_id not in _job_status_cache:
            return None
        cached_info, cached_at = _job_status_cache[job_id]
        age_seconds = (datetime.utcnow() - cached_at).total_seconds()
        if age_seconds > JOB_STATUS_CACHE_TTL_SECONDS:
            del _job_status_cache[job_id]
            return None
        return cached_info


def _set_cached_job_status(job_id: str, job_info: BulkJobInfo) -> None:
    """Set job status in cache."""
    with _job_cache_lock:
        _job_status_cache[job_id] = (job_info, datetime.utcnow())


def _invalidate_job_cache(job_id: str) -> None:
    """Invalidate cache for a specific job."""
    with _job_cache_lock:
        if job_id in _job_status_cache:
            del _job_status_cache[job_id]


def _clear_job_cache() -> None:
    """Clear all cached job statuses."""
    with _job_cache_lock:
        _job_status_cache.clear()


class BulkJobService(ThreadSafeSingleton):
    """
    Service for orchestrating bulk document processing.

    Responsibilities:
    - Create and validate bulk jobs
    - Coordinate document processing via LangGraph
    - Track progress and update job status
    - Handle concurrency and rate limiting
    """

    def _initialize(self) -> None:
        """Initialize service resources."""
        self.config = get_bulk_config()
        self._folder_manager = get_folder_manager()
        logger.info("BulkJobService initialized")

    async def create_job(
        self,
        org_id: str,
        folder_name: str,
        options: Optional[ProcessingOptions] = None,
    ) -> BulkJobInfo:
        """
        Create a new bulk processing job.

        Args:
            org_id: Organization ID
            folder_name: Name of the bulk folder
            options: Processing options (uses defaults if not provided)

        Returns:
            Created job info

        Raises:
            ValueError: If folder doesn't exist or exceeds limits
        """
        # Validate folder exists and meets requirements
        is_valid, message = await self._folder_manager.validate_folder_limit(
            org_id, folder_name
        )
        if not is_valid:
            raise ValueError(message)

        # Get folder info
        folder_info = await self._folder_manager.get_folder_info(org_id, folder_name)
        if not folder_info:
            raise ValueError(f"Folder not found: {folder_name}")

        # Check quota before creating job
        await self._check_quota(org_id, folder_info.document_count)

        # Use default options if not provided
        if options is None:
            options = ProcessingOptions(
                generate_summary=self.config.default_generate_summary,
                generate_faqs=self.config.default_generate_faqs,
                generate_questions=self.config.default_generate_questions,
                num_faqs=self.config.default_num_faqs,
                num_questions=self.config.default_num_questions,
                summary_max_words=self.config.default_summary_max_words,
            )

        # Create job record
        job_dict = await bulk_repository.create_bulk_job(
            organization_id=org_id,
            folder_name=folder_name,
            source_path=folder_info.gcs_path,
            total_documents=folder_info.document_count,
            options=options.model_dump(),
        )

        # Create document items
        documents = await self._folder_manager.list_documents(org_id, folder_name)
        for doc_path in documents:
            filename = doc_path.split("/")[-1]
            await bulk_repository.create_document_item(
                bulk_job_id=job_dict["id"],
                original_path=doc_path,
                original_filename=filename,
            )

        logger.info(
            f"Created bulk job {job_dict['id']} for folder '{folder_name}' "
            f"with {folder_info.document_count} documents"
        )

        return BulkJobInfo.from_dict(job_dict)

    async def start_job_processing(self, job_id: str) -> None:
        """
        Start processing all documents in a job.

        Called by the queue when a 'start' event is received.
        """
        job_dict = await bulk_repository.get_bulk_job(job_id)
        if not job_dict:
            logger.error(f"Job not found: {job_id}")
            return

        if job_dict["status"] not in ["pending"]:
            logger.warning(f"Job {job_id} is not pending, status={job_dict['status']}")
            return

        # Update job status to processing
        await bulk_repository.update_bulk_job_status(
            job_id,
            status=BulkJobStatus.PROCESSING.value,
            started_at=datetime.utcnow(),
        )
        _invalidate_job_cache(job_id)

        # Get pending documents and queue them
        pending_docs = await bulk_repository.get_pending_documents(
            job_id,
            limit=self.config.concurrent_documents,
        )

        from .queue import get_bulk_queue

        queue = get_bulk_queue()
        for doc in pending_docs:
            queue.enqueue(BulkJobEvent(
                job_id=job_id,
                action="process_document",
                document_id=doc["id"],
            ))

        logger.info(f"Started job {job_id}, queued {len(pending_docs)} documents")

    async def process_single_document(
        self,
        job_id: str,
        document_id: str,
    ) -> None:
        """
        Process a single document through the LangGraph workflow.

        Called by the queue when a 'process_document' event is received.
        """
        job_dict = await bulk_repository.get_bulk_job(job_id)
        doc_dict = await bulk_repository.get_document_item(document_id)

        if not job_dict or not doc_dict:
            logger.error(f"Job or document not found: job={job_id}, doc={document_id}")
            return

        # Check if job was cancelled - stop processing immediately
        if job_dict["status"] == "cancelled":
            logger.info(f"Job {job_id} was cancelled, skipping document {document_id}")
            return

        # Check if job is still processing
        if job_dict["status"] not in ["processing"]:
            logger.warning(f"Job {job_id} is not processing (status={job_dict['status']}), skipping document")
            return

        # Get processing options
        options = ProcessingOptions(**job_dict.get("options", {}))

        # Extract org_name from source_path (gs://bucket/org_name/original/folder)
        source_path = job_dict.get("source_path", "")
        path_parts = extract_gcs_path_parts(source_path)
        # path_parts[0] = bucket, path_parts[1] = org_name
        org_name = path_parts[1] if len(path_parts) > 1 else job_dict["organization_id"]

        # Create initial state
        state = create_initial_state(
            document_id=document_id,
            bulk_job_id=job_id,
            original_path=doc_dict["original_path"],
            org_id=job_dict["organization_id"],
            org_name=org_name,
            folder_name=job_dict["folder_name"],
            options=options,
        )

        # Run document through LangGraph
        graph = get_document_graph()
        config = {"configurable": {"thread_id": document_id}}

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                get_executors().agent_executor,
                lambda: asyncio.run(self._run_graph(graph, state, config)),
            )

            # Update job counters based on result
            final_status = result.get("status", "failed")
            if final_status == DocumentItemStatus.COMPLETED.value:
                await bulk_repository.increment_job_completed(
                    job_id,
                    token_usage=result.get("token_usage", 0),
                    llamaparse_pages=result.get("llamaparse_pages", 0),
                )
            elif final_status == DocumentItemStatus.SKIPPED.value:
                await bulk_repository.increment_job_skipped(job_id)
            else:
                await bulk_repository.increment_job_failed(job_id)

            logger.info(
                f"Processed document {document_id}: status={final_status}, "
                f"time={result.get('total_time_ms', 0)}ms"
            )

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            await bulk_repository.update_document_item(
                document_id,
                status=DocumentItemStatus.FAILED.value,
                error_message=str(e),
            )
            await bulk_repository.increment_job_failed(job_id)

        # Queue next documents
        await self._queue_next_documents(job_id)

    async def _run_graph(self, graph, state, config):
        """Run the LangGraph workflow (helper for executor)."""
        result = await graph.ainvoke(state, config)
        return result

    async def _queue_next_documents(self, job_id: str) -> None:
        """Check progress and queue more documents if needed."""
        job_dict = await bulk_repository.get_bulk_job(job_id)
        if not job_dict:
            return

        # Check if job is complete
        processed = (
            job_dict["completed_count"] +
            job_dict["failed_count"] +
            job_dict["skipped_count"]
        )

        if processed >= job_dict["total_documents"]:
            # All documents processed, finalize job
            from .queue import get_bulk_queue

            get_bulk_queue().enqueue(BulkJobEvent(
                job_id=job_id,
                action="complete",
            ))
            return

        # Get currently processing count
        in_progress = await bulk_repository.count_in_progress_documents(job_id)
        slots_available = self.config.concurrent_documents - in_progress

        if slots_available <= 0:
            return

        # Queue more documents
        pending_docs = await bulk_repository.get_pending_documents(
            job_id,
            limit=slots_available,
        )

        from .queue import get_bulk_queue

        queue = get_bulk_queue()
        for doc in pending_docs:
            queue.enqueue(BulkJobEvent(
                job_id=job_id,
                action="process_document",
                document_id=doc["id"],
            ))

    async def finalize_job(self, job_id: str) -> None:
        """
        Finalize a completed bulk job.

        Called when all documents have been processed.
        """
        job_dict = await bulk_repository.get_bulk_job(job_id)
        if not job_dict:
            logger.error(f"Job not found for finalization: {job_id}")
            return

        # Determine final status
        if job_dict["failed_count"] == 0:
            final_status = BulkJobStatus.COMPLETED.value
        elif job_dict["completed_count"] > 0:
            final_status = BulkJobStatus.PARTIAL_FAILURE.value
        else:
            final_status = BulkJobStatus.FAILED.value

        await bulk_repository.update_bulk_job_status(
            job_id,
            status=final_status,
            completed_at=datetime.utcnow(),
        )
        _invalidate_job_cache(job_id)

        logger.info(
            f"Finalized job {job_id}: status={final_status}, "
            f"completed={job_dict['completed_count']}, "
            f"failed={job_dict['failed_count']}, "
            f"skipped={job_dict['skipped_count']}"
        )

    async def cancel_job(self, job_id: str) -> None:
        """
        Cancel a running bulk job atomically.

        1. Marks job as CANCELLED first (stops new document processing)
        2. Marks ALL cancellable documents as SKIPPED in a single atomic operation
        3. Updates the skipped count
        """
        job_dict = await bulk_repository.get_bulk_job(job_id)
        if not job_dict:
            logger.error(f"Job not found for cancellation: {job_id}")
            return

        if job_dict["status"] not in ["pending", "processing"]:
            logger.warning(f"Job {job_id} cannot be cancelled, status={job_dict['status']}")
            return

        # 1. Mark job as CANCELLED FIRST - this stops new document processing
        await bulk_repository.update_bulk_job_status(
            job_id,
            status=BulkJobStatus.CANCELLED.value,
            completed_at=datetime.utcnow(),
        )
        _invalidate_job_cache(job_id)

        # 2. Cancel ALL documents atomically in a single SQL UPDATE
        cancelled_count = await bulk_repository.cancel_all_documents(job_id)

        # 3. Update the skipped count in one operation
        await bulk_repository.set_job_skipped_count(job_id, cancelled_count)

        logger.info(f"Cancelled job {job_id} atomically, skipped {cancelled_count} documents")

    async def retry_failed_documents(
        self,
        job_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> int:
        """
        Retry failed documents in a bulk job.

        Args:
            job_id: Job ID
            document_ids: Specific documents to retry (all failed if None)

        Returns:
            Number of documents queued for retry
        """
        job_dict = await bulk_repository.get_bulk_job(job_id)
        if not job_dict:
            raise ValueError(f"Job not found: {job_id}")

        if document_ids:
            # Retry specific documents
            count = 0
            for doc_id in document_ids:
                doc = await bulk_repository.get_document_item(doc_id)
                if doc and doc["status"] == DocumentItemStatus.FAILED.value:
                    if doc["retry_count"] < self.config.max_retries_per_document:
                        await bulk_repository.reset_document_for_retry(doc_id)
                        count += 1
        else:
            # Retry all failed documents
            count = await bulk_repository.bulk_reset_failed_documents(job_id)

        if count > 0:
            # Update job status if needed
            if job_dict["status"] in ["completed", "partial_failure", "failed"]:
                await bulk_repository.update_bulk_job_status(
                    job_id,
                    status=BulkJobStatus.PROCESSING.value,
                )
                _invalidate_job_cache(job_id)

            # Queue documents for processing
            pending_docs = await bulk_repository.get_pending_documents(
                job_id,
                limit=self.config.concurrent_documents,
            )

            from .queue import get_bulk_queue

            queue = get_bulk_queue()
            for doc in pending_docs:
                queue.enqueue(BulkJobEvent(
                    job_id=job_id,
                    action="process_document",
                    document_id=doc["id"],
                ))

        logger.info(f"Queued {count} documents for retry in job {job_id}")
        return count

    async def get_job_status(
        self,
        job_id: str,
        include_documents: bool = False,
    ) -> Optional[BulkJobInfo]:
        """
        Get detailed status of a bulk job.

        Uses in-memory cache with short TTL to reduce database polling load.
        Cache is invalidated when job status changes.

        Args:
            job_id: Job ID
            include_documents: Include document-level details

        Returns:
            Job info with optional document details
        """
        # Check cache first (only for requests without documents)
        # Document-level details change frequently so we don't cache those
        if not include_documents:
            cached = _get_cached_job_status(job_id)
            if cached is not None:
                return cached

        # Fetch from database
        job_dict = await bulk_repository.get_bulk_job(job_id)
        if not job_dict:
            return None

        job_info = BulkJobInfo.from_dict(job_dict)

        if include_documents:
            doc_dicts = await bulk_repository.get_all_document_items(job_id)
            job_info.documents = [
                DocumentItemInfo.from_dict(d) for d in doc_dicts
            ]
        else:
            # Cache the job info (without documents)
            _set_cached_job_status(job_id, job_info)

        return job_info

    async def _check_quota(self, org_id: str, document_count: int) -> None:
        """
        Check quota before creating a job.

        Raises:
            ValueError: If quota would be exceeded
        """
        try:
            from src.core.usage import get_quota_checker

            checker = get_quota_checker()

            # Check LlamaParse pages (1 per document)
            pages_status = await checker.check_quota(
                org_id, "llamaparse_pages", document_count
            )
            if not pages_status.allowed:
                raise ValueError(f"LlamaParse quota exceeded: {pages_status.upgrade_message}")

            # Estimate tokens (rough estimate: 2000 per document)
            estimated_tokens = document_count * 2000
            token_status = await checker.check_quota(
                org_id, "tokens", estimated_tokens
            )
            if not token_status.allowed:
                raise ValueError(f"Token quota exceeded: {token_status.upgrade_message}")

        except ImportError:
            # Quota checker not available, skip check
            logger.warning("Quota checker not available, skipping quota check")
        except Exception as e:
            # Log but don't block on quota check errors
            logger.warning(f"Quota check failed: {e}")


# Singleton instance
_bulk_service: Optional[BulkJobService] = None


def get_bulk_service() -> BulkJobService:
    """Get the bulk job service singleton."""
    global _bulk_service
    if _bulk_service is None:
        _bulk_service = BulkJobService.get_instance()
    return _bulk_service


def reset_bulk_service() -> None:
    """Reset the service singleton (for testing)."""
    global _bulk_service
    _bulk_service = None
