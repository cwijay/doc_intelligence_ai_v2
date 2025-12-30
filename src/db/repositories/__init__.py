"""
Repository layer for database operations.

Provides:
- audit_repository: Document tracking, job management, audit logging
- memory_repository: Long-term memory for user preferences and summaries
- rag_repository: File search stores and document folders for RAG
- extraction_repository: Dynamic table creation and data persistence for extracted data
"""

from .audit_repository import (
    # Core operations
    get_file_hash,
    register_document,
    find_cached_result,
    start_job,
    complete_job,
    fail_job,
    log_event,
    get_processing_history,
    # Dashboard queries
    get_document_by_name,
    get_jobs_by_document,
    get_audit_trail,
    get_document_summary,
    # Document generations
    save_document_generation,
    find_cached_generation,
    get_generations_by_document,
    get_recent_generations,
)

from .memory_repository import PostgresLongTermMemory

from . import rag_repository

from .extraction_repository import (
    ensure_extraction_tables_exist,
    save_extracted_record,
    get_extracted_records,
    get_extracted_record_with_line_items,
    get_record_count,
    delete_extracted_record,
    check_table_exists,
    get_organization_name,
)

__all__ = [
    # Audit operations
    "get_file_hash",
    "register_document",
    "find_cached_result",
    "start_job",
    "complete_job",
    "fail_job",
    "log_event",
    "get_processing_history",
    "get_document_by_name",
    "get_jobs_by_document",
    "get_audit_trail",
    "get_document_summary",
    "save_document_generation",
    "find_cached_generation",
    "get_generations_by_document",
    "get_recent_generations",
    # Memory
    "PostgresLongTermMemory",
    # RAG
    "rag_repository",
    # Extraction
    "ensure_extraction_tables_exist",
    "save_extracted_record",
    "get_extracted_records",
    "get_extracted_record_with_line_items",
    "get_record_count",
    "delete_extracted_record",
    "check_table_exists",
    "get_organization_name",
]
