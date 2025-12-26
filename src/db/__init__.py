"""
PostgreSQL database package for Document Intelligence AI.

Provides:
- SQLAlchemy 2.0 async ORM models (re-exported from biz2bricks_core)
- Connection management with Cloud SQL connector
- Repository layer for audit and memory operations

All models are now defined in biz2bricks_core for sharing across applications.
Backwards-compatible aliases are provided for existing code.
"""

from .connection import DatabaseManager, db, get_session

# Re-export all models from biz2bricks_core
from biz2bricks_core import (
    Base,
    # Core models
    OrganizationModel,
    UserModel,
    FolderModel,
    DocumentModel,
    AuditLogModel,
    AuditAction,
    AuditEntityType,
    # AI processing models
    ProcessingJobModel,
    DocumentGenerationModel,
    UserPreferenceModel,
    ConversationSummaryModel,
    MemoryEntryModel,
    FileSearchStoreModel,
    DocumentFolderModel,
)

# Backwards-compatible aliases from models.py
from .models import (
    Document,
    AuditLog,
    ProcessingJob,
    DocumentGeneration,
    UserPreference,
    ConversationSummary,
    MemoryEntry,
    FileSearchStore,
    DocumentFolder,
)

__all__ = [
    # Connection management
    "DatabaseManager",
    "db",
    "get_session",
    # Base
    "Base",
    # Core models
    "OrganizationModel",
    "UserModel",
    "FolderModel",
    "DocumentModel",
    "AuditLogModel",
    "AuditAction",
    "AuditEntityType",
    # AI processing models
    "ProcessingJobModel",
    "DocumentGenerationModel",
    "UserPreferenceModel",
    "ConversationSummaryModel",
    "MemoryEntryModel",
    "FileSearchStoreModel",
    "DocumentFolderModel",
    # Backwards-compatible aliases
    "Document",
    "AuditLog",
    "ProcessingJob",
    "DocumentGeneration",
    "UserPreference",
    "ConversationSummary",
    "MemoryEntry",
    "FileSearchStore",
    "DocumentFolder",
]
