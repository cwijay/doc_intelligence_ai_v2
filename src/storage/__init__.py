"""Storage module for document persistence in GCS."""

from .base import StorageBackend
from .gcs import GCSStorage
from .config import StorageConfig, get_storage, get_storage_config

__all__ = [
    "StorageBackend",
    "GCSStorage",
    "StorageConfig",
    "get_storage",
    "get_storage_config",
]
