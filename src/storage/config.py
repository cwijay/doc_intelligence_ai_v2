"""Storage configuration and factory."""

import os
import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from .base import StorageBackend

logger = logging.getLogger(__name__)


class StorageConfig(BaseModel):
    """Storage configuration from environment variables."""

    gcs_bucket: str = Field(
        default_factory=lambda: os.getenv(
            "GCS_BUCKET", "biz2bricks-dev-v1-document-store"
        ),
        description="GCS bucket name",
    )

    gcs_prefix: str = Field(
        default_factory=lambda: os.getenv("GCS_PREFIX", ""),
        description="Prefix/folder within GCS bucket (empty by default for multi-tenant org-based paths)",
    )

    parsed_directory: str = Field(
        default_factory=lambda: os.getenv("PARSED_DIRECTORY", "parsed"),
        description="Directory for parsed markdown files",
    )

    generated_directory: str = Field(
        default_factory=lambda: os.getenv("GENERATED_DIRECTORY", "generated"),
        description="Directory for generated content (summaries, FAQs, etc.)",
    )

    @field_validator("gcs_bucket")
    @classmethod
    def validate_bucket(cls, v: str) -> str:
        if not v:
            raise ValueError("GCS_BUCKET must be set")
        return v


# Global storage instance (singleton)
_storage: Optional[StorageBackend] = None
_config: Optional[StorageConfig] = None


def get_storage() -> StorageBackend:
    """
    Get or create the storage backend singleton.

    Returns:
        StorageBackend: Configured GCS storage instance
    """
    global _storage, _config

    if _storage is None:
        from .gcs import GCSStorage

        _config = StorageConfig()

        _storage = GCSStorage(
            bucket_name=_config.gcs_bucket,
            prefix=_config.gcs_prefix,
        )
        logger.info(
            f"Storage initialized: gs://{_config.gcs_bucket}/{_config.gcs_prefix}"
        )

    return _storage


def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    global _config
    if _config is None:
        _config = StorageConfig()
    return _config


def reset_storage() -> None:
    """Reset storage singleton (for testing)."""
    global _storage, _config
    _storage = None
    _config = None
