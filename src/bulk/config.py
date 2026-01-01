"""
Bulk processing configuration.

All settings are configurable via environment variables with BULK_ prefix.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

from src.utils.env_utils import parse_bool_env, parse_int_env, parse_float_env


class BulkProcessingConfig(BaseSettings):
    """Configuration for bulk document processing."""

    # Folder and document limits
    max_documents_per_folder: int = Field(
        default=10,
        description="Maximum documents allowed per bulk folder",
    )
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file size in MB",
    )
    supported_extensions: list[str] = Field(
        default=[
            ".pdf", ".doc", ".docx", ".ppt", ".pptx",
            ".txt", ".rtf", ".xlsx", ".xls", ".csv",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
            ".html", ".htm",
        ],
        description="Supported file extensions for processing",
    )

    # Processing concurrency
    concurrent_documents: int = Field(
        default=3,
        description="Number of documents to process concurrently",
    )

    # Timeouts (in seconds)
    parse_timeout_seconds: int = Field(
        default=300,
        description="Timeout for parsing a single document",
    )
    index_timeout_seconds: int = Field(
        default=60,
        description="Timeout for indexing a single document",
    )
    generation_timeout_seconds: int = Field(
        default=300,
        description="Timeout for generating content for a single document",
    )
    job_timeout_seconds: int = Field(
        default=3600,
        description="Overall timeout for a bulk job (1 hour)",
    )

    # Retry settings
    max_retries_per_document: int = Field(
        default=3,
        description="Maximum retry attempts per document",
    )
    retry_delay_seconds: int = Field(
        default=30,
        description="Base delay between retries",
    )

    # Auto-start settings (for Cloud Function triggers)
    auto_start_delay_seconds: int = Field(
        default=60,
        description="Wait time after last upload before auto-starting job",
    )
    auto_start_min_documents: int = Field(
        default=1,
        description="Minimum documents required to start processing",
    )

    # Queue settings
    queue_max_size: int = Field(
        default=100,
        description="Maximum size of the processing queue",
    )
    worker_poll_interval_seconds: float = Field(
        default=1.0,
        description="Interval for queue workers to poll for new events",
    )

    # GCS settings
    bulk_folder_prefix: str = Field(
        default="bulk",
        description="Prefix for bulk upload folders (e.g., org/bulk/folder)",
    )
    signed_url_expiration_minutes: int = Field(
        default=60,
        description="Expiration time for signed upload URLs",
    )

    # Webhook security
    webhook_secret: Optional[str] = Field(
        default=None,
        description="Shared secret for webhook authentication",
    )
    webhook_enabled: bool = Field(
        default=True,
        description="Enable webhook endpoint for Cloud Function triggers",
    )

    # Content generation defaults
    default_generate_summary: bool = Field(
        default=True,
        description="Generate summary by default",
    )
    default_generate_faqs: bool = Field(
        default=True,
        description="Generate FAQs by default",
    )
    default_generate_questions: bool = Field(
        default=True,
        description="Generate questions by default",
    )
    default_num_faqs: int = Field(
        default=10,
        description="Default number of FAQs to generate",
    )
    default_num_questions: int = Field(
        default=10,
        description="Default number of questions to generate",
    )
    default_summary_max_words: int = Field(
        default=500,
        description="Default maximum words for summary",
    )

    # LangGraph checkpointing
    use_postgres_checkpointer: bool = Field(
        default=True,
        description="Use PostgreSQL for LangGraph checkpointing (durable)",
    )

    class Config:
        env_prefix = "BULK_"
        case_sensitive = False

    @classmethod
    def from_env(cls) -> "BulkProcessingConfig":
        """Create config from environment variables."""
        return cls(
            max_documents_per_folder=parse_int_env("BULK_MAX_DOCUMENTS_PER_FOLDER", 10),
            max_file_size_mb=parse_int_env("BULK_MAX_FILE_SIZE_MB", 50),
            concurrent_documents=parse_int_env("BULK_CONCURRENT_DOCUMENTS", 3),
            parse_timeout_seconds=parse_int_env("BULK_PARSE_TIMEOUT_SECONDS", 300),
            index_timeout_seconds=parse_int_env("BULK_INDEX_TIMEOUT_SECONDS", 60),
            generation_timeout_seconds=parse_int_env("BULK_GENERATION_TIMEOUT_SECONDS", 300),
            job_timeout_seconds=parse_int_env("BULK_JOB_TIMEOUT_SECONDS", 3600),
            max_retries_per_document=parse_int_env("BULK_MAX_RETRIES_PER_DOCUMENT", 3),
            retry_delay_seconds=parse_int_env("BULK_RETRY_DELAY_SECONDS", 30),
            auto_start_delay_seconds=parse_int_env("BULK_AUTO_START_DELAY_SECONDS", 60),
            auto_start_min_documents=parse_int_env("BULK_AUTO_START_MIN_DOCUMENTS", 1),
            queue_max_size=parse_int_env("BULK_QUEUE_MAX_SIZE", 100),
            worker_poll_interval_seconds=parse_float_env("BULK_WORKER_POLL_INTERVAL_SECONDS", 1.0),
            bulk_folder_prefix=os.getenv("BULK_FOLDER_PREFIX", "bulk"),
            signed_url_expiration_minutes=parse_int_env("BULK_SIGNED_URL_EXPIRATION_MINUTES", 60),
            webhook_secret=os.getenv("BULK_WEBHOOK_SECRET"),
            webhook_enabled=parse_bool_env("BULK_WEBHOOK_ENABLED", True),
            default_generate_summary=parse_bool_env("BULK_DEFAULT_GENERATE_SUMMARY", True),
            default_generate_faqs=parse_bool_env("BULK_DEFAULT_GENERATE_FAQS", True),
            default_generate_questions=parse_bool_env("BULK_DEFAULT_GENERATE_QUESTIONS", True),
            default_num_faqs=parse_int_env("BULK_DEFAULT_NUM_FAQS", 10),
            default_num_questions=parse_int_env("BULK_DEFAULT_NUM_QUESTIONS", 10),
            default_summary_max_words=parse_int_env("BULK_DEFAULT_SUMMARY_MAX_WORDS", 500),
            use_postgres_checkpointer=parse_bool_env("BULK_USE_POSTGRES_CHECKPOINTER", True),
        )


# Singleton config instance
_config: Optional[BulkProcessingConfig] = None


def get_bulk_config() -> BulkProcessingConfig:
    """Get the bulk processing config singleton."""
    global _config
    if _config is None:
        _config = BulkProcessingConfig.from_env()
    return _config


def reset_bulk_config() -> None:
    """Reset the config singleton (for testing)."""
    global _config
    _config = None
