"""Unit tests for StorageConfig and factory functions."""

import pytest
from unittest.mock import patch


class TestStorageConfig:
    """Tests for StorageConfig class."""

    def test_default_values(self, clean_env):
        """Test default values when env vars not set."""
        from src.storage.config import StorageConfig

        config = StorageConfig()

        assert config.gcs_bucket == "biz2bricks-dev-v1-document-store"
        assert config.gcs_prefix == ""  # Empty for multi-tenant org-based paths
        assert config.parsed_directory == "parsed"
        assert config.generated_directory == "generated"

    def test_values_from_env(self, monkeypatch):
        """Test values from environment variables."""
        monkeypatch.setenv("GCS_BUCKET", "custom-bucket")
        monkeypatch.setenv("GCS_PREFIX", "custom-prefix")
        monkeypatch.setenv("PARSED_DIRECTORY", "custom-parsed")
        monkeypatch.setenv("GENERATED_DIRECTORY", "custom-generated")

        from src.storage.config import StorageConfig

        config = StorageConfig()

        assert config.gcs_bucket == "custom-bucket"
        assert config.gcs_prefix == "custom-prefix"
        assert config.parsed_directory == "custom-parsed"
        assert config.generated_directory == "custom-generated"

    def test_default_bucket_used_when_env_empty(self, monkeypatch):
        """Test that default bucket is used when env is empty or unset."""
        # When GCS_BUCKET env var is not set, default should be used
        monkeypatch.delenv("GCS_BUCKET", raising=False)

        from src.storage.config import StorageConfig

        config = StorageConfig()
        assert config.gcs_bucket == "biz2bricks-dev-v1-document-store"


class TestGetStorage:
    """Tests for get_storage factory function."""

    def test_get_storage_returns_gcs_storage(self, mock_storage_client, mock_gcs_env):
        """Test that get_storage returns a GCSStorage instance."""
        from src.storage.config import get_storage
        from src.storage.gcs import GCSStorage

        storage = get_storage()

        assert isinstance(storage, GCSStorage)

    def test_get_storage_singleton(self, mock_storage_client, mock_gcs_env):
        """Test that get_storage returns the same instance."""
        from src.storage.config import get_storage

        storage1 = get_storage()
        storage2 = get_storage()

        assert storage1 is storage2


class TestGetStorageConfig:
    """Tests for get_storage_config function."""

    def test_get_storage_config_returns_config(self, mock_gcs_env):
        """Test that get_storage_config returns a StorageConfig."""
        from src.storage.config import get_storage_config, StorageConfig

        config = get_storage_config()

        assert isinstance(config, StorageConfig)

    def test_get_storage_config_caches(self, mock_gcs_env):
        """Test that config is cached after first call."""
        from src.storage.config import get_storage_config

        config1 = get_storage_config()
        config2 = get_storage_config()

        assert config1 is config2


class TestResetStorage:
    """Tests for reset_storage function."""

    def test_reset_storage_clears_singleton(self, mock_storage_client, mock_gcs_env):
        """Test that reset_storage clears the singleton."""
        from src.storage.config import get_storage, reset_storage

        storage1 = get_storage()
        reset_storage()
        storage2 = get_storage()

        # After reset, should create new instance
        assert storage1 is not storage2

    def test_reset_storage_clears_config(self, mock_gcs_env, monkeypatch):
        """Test that reset_storage clears the config."""
        from src.storage.config import get_storage_config, reset_storage

        config1 = get_storage_config()

        # Change env and reset
        monkeypatch.setenv("GCS_BUCKET", "new-bucket")
        reset_storage()

        config2 = get_storage_config()

        assert config1 is not config2
        assert config2.gcs_bucket == "new-bucket"
