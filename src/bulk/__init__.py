"""
Bulk document processing module.

Provides automated bulk document processing with:
- Folder management with signed URLs for direct GCS upload
- LangGraph StateGraph for per-document workflow orchestration
- Background job queue for async processing
- Cloud Function webhook for auto-triggering on upload
- Content generation (summary, FAQs, questions) for each document
"""

from .config import BulkProcessingConfig, get_bulk_config
from .schemas import (
    BulkJobStatus,
    DocumentItemStatus,
    BulkFolderInfo,
    BulkJobInfo,
    DocumentItemInfo,
    ProcessingOptions,
)

__all__ = [
    # Config
    "BulkProcessingConfig",
    "get_bulk_config",
    # Schemas
    "BulkJobStatus",
    "DocumentItemStatus",
    "BulkFolderInfo",
    "BulkJobInfo",
    "DocumentItemInfo",
    "ProcessingOptions",
]
