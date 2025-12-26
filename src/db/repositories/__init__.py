"""
Repository layer for database operations.

Provides:
- audit_repository: Document tracking, job management, audit logging
- memory_repository: Long-term memory for user preferences and summaries
- rag_repository: File search stores and document folders for RAG
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
]
