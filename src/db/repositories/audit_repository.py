"""
DEPRECATED: This module has been split into focused repositories.

This file re-exports all functions from the split modules for backwards compatibility.
New code should import directly from the audit subpackage:

    # New way (preferred):
    from src.db.repositories.audit import (
        get_file_hash,
        register_document,
        start_job,
        log_event,
        save_document_generation,
    )

    # Or import specific repositories:
    from src.db.repositories.audit.document_repository import get_document_by_name
    from src.db.repositories.audit.job_repository import start_job
    from src.db.repositories.audit.audit_log_repository import log_event
    from src.db.repositories.audit.generation_repository import save_document_generation

Split modules:
- audit.document_repository: Document CRUD operations
- audit.job_repository: Processing job lifecycle management
- audit.audit_log_repository: Event logging and audit trail queries
- audit.generation_repository: Document generation CRUD operations
"""

import warnings

# Emit deprecation warning on import
warnings.warn(
    "audit_repository module is deprecated. "
    "Import from src.db.repositories.audit instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all functions for backwards compatibility
from .audit import (
    # Document operations
    get_file_hash,
    register_document,
    register_uploaded_document,
    update_document_status,
    register_or_update_parsed_document,
    get_document_by_path,
    get_document_by_name,
    list_documents_by_status,
    # Job operations
    find_cached_result,
    start_job,
    complete_job,
    fail_job,
    get_processing_history,
    get_jobs_by_document,
    # Audit log operations
    log_event,
    get_audit_trail,
    get_document_summary,
    # Generation operations
    save_document_generation,
    find_cached_generation,
    get_generations_by_document,
    get_recent_generations,
    delete_test_records,
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
]
