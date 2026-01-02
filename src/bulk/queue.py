"""
Bulk processing queue.

Background queue for processing bulk job events asynchronously.
Extends the core BackgroundQueue pattern for bulk document processing.
"""

import logging
from typing import Optional

from src.core.queues import BackgroundQueue

from .schemas import BulkJobEvent

logger = logging.getLogger(__name__)


class BulkJobQueue(BackgroundQueue[BulkJobEvent]):
    """
    Background queue for bulk document processing jobs.

    Handles events:
    - start: Begin processing all documents in a job
    - process_document: Process a single document
    - complete: Finalize a completed job
    - cancel: Cancel a running job

    Supports concurrent document processing controlled by BULK_CONCURRENT_DOCUMENTS.
    """

    def _get_queue_name(self) -> str:
        return "bulk-job-queue"

    def _get_max_concurrent(self) -> int:
        """Return max concurrent document processing tasks.

        Uses the bulk config's concurrent_documents setting.
        """
        from .config import get_bulk_config
        return get_bulk_config().concurrent_documents

    async def _process_event(self, event: BulkJobEvent) -> None:
        """Process a bulk job event."""
        from .service import get_bulk_service

        service = get_bulk_service()

        logger.debug(f"Processing bulk event: action={event.action}, job={event.job_id}")

        try:
            if event.action == "start":
                await service.start_job_processing(event.job_id)

            elif event.action == "process_document":
                if event.document_id:
                    await service.process_single_document(
                        event.job_id,
                        event.document_id,
                    )
                else:
                    logger.warning(f"process_document event missing document_id: {event}")

            elif event.action == "complete":
                await service.finalize_job(event.job_id)

            elif event.action == "cancel":
                await service.cancel_job(event.job_id)

            else:
                logger.warning(f"Unknown bulk event action: {event.action}")

        except Exception as e:
            logger.error(f"Error processing bulk event {event.action}: {e}")
            raise


# Singleton instance
_bulk_queue: Optional[BulkJobQueue] = None


def get_bulk_queue() -> BulkJobQueue:
    """Get the bulk job queue singleton."""
    global _bulk_queue
    if _bulk_queue is None:
        _bulk_queue = BulkJobQueue.get_instance()
    return _bulk_queue


def start_bulk_queue() -> None:
    """Start the bulk job queue (call at app startup)."""
    queue = get_bulk_queue()
    queue.start()
    logger.info("Bulk job queue started")


def stop_bulk_queue(wait: bool = True) -> None:
    """Stop the bulk job queue (call at app shutdown)."""
    global _bulk_queue
    if _bulk_queue:
        _bulk_queue.shutdown(wait=wait)
        _bulk_queue = None
        logger.info("Bulk job queue stopped")
