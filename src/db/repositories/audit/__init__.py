"""Audit repositories package - Split from audit_repository.py for maintainability.

This package contains focused repository modules:
- document_repository: Document CRUD operations
- job_repository: Processing job lifecycle management
- audit_log_repository: Event logging and audit trail queries
- generation_repository: Document generation CRUD operations

All functions are re-exported here for backwards compatibility.
"""

# Document operations
from .document_repository import (
    get_file_hash,
    register_document,
    register_uploaded_document,
    update_document_status,
    register_or_update_parsed_document,
    get_document_by_path,
    get_document_by_name,
    list_documents_by_status,
)

# Job operations
from .job_repository import (
    find_cached_result,
    start_job,
    complete_job,
    fail_job,
    get_processing_history,
    get_jobs_by_document,
)

# Audit log operations
from .audit_log_repository import (
    log_event,
    get_audit_trail,
    get_document_summary,
)

# Generation operations
from .generation_repository import (
    save_document_generation,
    find_cached_generation,
    get_generations_by_document,
    get_recent_generations,
    delete_test_records,
)

# Statistics operations
from .stats_repository import (
    get_dashboard_stats,
    get_job_by_id,
    count_jobs,
    count_documents,
    count_generations,
    count_audit_events,
)

__all__ = [
    # Document operations
    "get_file_hash",
    "register_document",
    "register_uploaded_document",
    "update_document_status",
    "register_or_update_parsed_document",
    "get_document_by_path",
    "get_document_by_name",
    "list_documents_by_status",
    # Job operations
    "find_cached_result",
    "start_job",
    "complete_job",
    "fail_job",
    "get_processing_history",
    "get_jobs_by_document",
    # Audit log operations
    "log_event",
    "get_audit_trail",
    "get_document_summary",
    # Generation operations
    "save_document_generation",
    "find_cached_generation",
    "get_generations_by_document",
    "get_recent_generations",
    "delete_test_records",
    # Statistics operations
    "get_dashboard_stats",
    "get_job_by_id",
    "count_jobs",
    "count_documents",
    "count_generations",
    "count_audit_events",
]
