"""
SQLAlchemy models re-exported from biz2bricks_core.

All database models are defined in the shared biz2bricks_core package.
This file provides backwards-compatible imports for existing code.

Core models:
- OrganizationModel, UserModel, FolderModel
- DocumentModel (with AI processing columns: file_hash, parsed_path, parsed_at)
- AuditLogModel (with AI audit columns: event_type, document_hash, file_name, job_id)

AI processing models:
- ProcessingJobModel: Document processing job tracking
- DocumentGenerationModel: Generated content cache (summaries, FAQs, questions)
- UserPreferenceModel: User preferences for long-term memory
- ConversationSummaryModel: Conversation summaries for memory
- MemoryEntryModel: Generic key-value memory storage
- FileSearchStoreModel: Gemini File Search store registry
- DocumentFolderModel: Document folder hierarchy for RAG

Usage tracking models:
- SubscriptionTierModel: Admin-editable tier configuration
- OrganizationSubscriptionModel: Per-org subscription state and usage counters
- TokenUsageRecordModel: Granular token usage logs for analytics
- ResourceUsageRecordModel: Non-token resource tracking (LlamaParse, file search)
- UsageAggregationModel: Pre-computed rollups for dashboards

RAG cache models:
- RAGQueryCacheModel: Semantic caching for RAG queries with pgvector
"""

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
    # Usage tracking models
    SubscriptionTierModel,
    OrganizationSubscriptionModel,
    TokenUsageRecordModel,
    ResourceUsageRecordModel,
    UsageAggregationModel,
    SubscriptionTier,
    OrganizationSubscription,
    TokenUsageRecord,
    ResourceUsageRecord,
    UsageAggregation,
    # RAG cache models
    RAGQueryCacheModel,
    RAGQueryCache,
    PGVECTOR_AVAILABLE,
)

# Backwards-compatible aliases for existing code
Document = DocumentModel
AuditLog = AuditLogModel
ProcessingJob = ProcessingJobModel
DocumentGeneration = DocumentGenerationModel
UserPreference = UserPreferenceModel
ConversationSummary = ConversationSummaryModel
MemoryEntry = MemoryEntryModel
FileSearchStore = FileSearchStoreModel
DocumentFolder = DocumentFolderModel

__all__ = [
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
    # Local models
    "RAGQueryCacheModel",
    "PGVECTOR_AVAILABLE",
    # Usage tracking models
    "SubscriptionTierModel",
    "OrganizationSubscriptionModel",
    "TokenUsageRecordModel",
    "ResourceUsageRecordModel",
    "UsageAggregationModel",
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
    "RAGQueryCache",
    "SubscriptionTier",
    "OrganizationSubscription",
    "TokenUsageRecord",
    "ResourceUsageRecord",
    "UsageAggregation",
]
