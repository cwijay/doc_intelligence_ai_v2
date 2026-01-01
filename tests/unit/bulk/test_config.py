"""Unit tests for bulk processing configuration."""

import pytest
import os
from unittest.mock import patch

from src.bulk.config import (
    BulkProcessingConfig,
    get_bulk_config,
    reset_bulk_config,
)


class TestBulkProcessingConfigDefaults:
    """Tests for BulkProcessingConfig default values."""

    def test_default_max_documents_per_folder(self):
        """Test default max documents per folder is 10."""
        config = BulkProcessingConfig()
        assert config.max_documents_per_folder == 10

    def test_default_max_file_size_mb(self):
        """Test default max file size is 50 MB."""
        config = BulkProcessingConfig()
        assert config.max_file_size_mb == 50

    def test_default_concurrent_documents(self):
        """Test default concurrent documents is 3."""
        config = BulkProcessingConfig()
        assert config.concurrent_documents == 3

    def test_default_parse_timeout(self):
        """Test default parse timeout is 300 seconds."""
        config = BulkProcessingConfig()
        assert config.parse_timeout_seconds == 300

    def test_default_index_timeout(self):
        """Test default index timeout is 60 seconds."""
        config = BulkProcessingConfig()
        assert config.index_timeout_seconds == 60

    def test_default_generation_timeout(self):
        """Test default generation timeout is 300 seconds."""
        config = BulkProcessingConfig()
        assert config.generation_timeout_seconds == 300

    def test_default_job_timeout(self):
        """Test default job timeout is 3600 seconds (1 hour)."""
        config = BulkProcessingConfig()
        assert config.job_timeout_seconds == 3600

    def test_default_max_retries(self):
        """Test default max retries per document is 3."""
        config = BulkProcessingConfig()
        assert config.max_retries_per_document == 3

    def test_default_retry_delay(self):
        """Test default retry delay is 30 seconds."""
        config = BulkProcessingConfig()
        assert config.retry_delay_seconds == 30

    def test_default_auto_start_delay(self):
        """Test default auto start delay is 60 seconds."""
        config = BulkProcessingConfig()
        assert config.auto_start_delay_seconds == 60

    def test_default_auto_start_min_documents(self):
        """Test default auto start min documents is 1."""
        config = BulkProcessingConfig()
        assert config.auto_start_min_documents == 1

    def test_default_queue_max_size(self):
        """Test default queue max size is 100."""
        config = BulkProcessingConfig()
        assert config.queue_max_size == 100

    def test_default_worker_poll_interval(self):
        """Test default worker poll interval is 1.0 seconds."""
        config = BulkProcessingConfig()
        assert config.worker_poll_interval_seconds == 1.0

    def test_default_bulk_folder_prefix(self):
        """Test default bulk folder prefix is 'bulk'."""
        config = BulkProcessingConfig()
        assert config.bulk_folder_prefix == "bulk"

    def test_default_signed_url_expiration(self):
        """Test default signed URL expiration is 60 minutes."""
        config = BulkProcessingConfig()
        assert config.signed_url_expiration_minutes == 60

    def test_default_webhook_enabled(self):
        """Test webhook is enabled by default."""
        config = BulkProcessingConfig()
        assert config.webhook_enabled is True

    def test_default_webhook_secret_none(self):
        """Test webhook secret is None by default."""
        config = BulkProcessingConfig()
        assert config.webhook_secret is None

    def test_default_generate_summary(self):
        """Test generate summary is True by default."""
        config = BulkProcessingConfig()
        assert config.default_generate_summary is True

    def test_default_generate_faqs(self):
        """Test generate FAQs is True by default."""
        config = BulkProcessingConfig()
        assert config.default_generate_faqs is True

    def test_default_generate_questions(self):
        """Test generate questions is True by default."""
        config = BulkProcessingConfig()
        assert config.default_generate_questions is True

    def test_default_num_faqs(self):
        """Test default num FAQs is 10."""
        config = BulkProcessingConfig()
        assert config.default_num_faqs == 10

    def test_default_num_questions(self):
        """Test default num questions is 10."""
        config = BulkProcessingConfig()
        assert config.default_num_questions == 10

    def test_default_summary_max_words(self):
        """Test default summary max words is 500."""
        config = BulkProcessingConfig()
        assert config.default_summary_max_words == 500

    def test_default_use_postgres_checkpointer(self):
        """Test use postgres checkpointer is True by default."""
        config = BulkProcessingConfig()
        assert config.use_postgres_checkpointer is True


class TestBulkProcessingConfigSupportedExtensions:
    """Tests for supported file extensions."""

    def test_pdf_supported(self):
        """Test PDF extension is supported."""
        config = BulkProcessingConfig()
        assert ".pdf" in config.supported_extensions

    def test_docx_supported(self):
        """Test DOCX extension is supported."""
        config = BulkProcessingConfig()
        assert ".docx" in config.supported_extensions

    def test_doc_supported(self):
        """Test DOC extension is supported."""
        config = BulkProcessingConfig()
        assert ".doc" in config.supported_extensions

    def test_pptx_supported(self):
        """Test PPTX extension is supported."""
        config = BulkProcessingConfig()
        assert ".pptx" in config.supported_extensions

    def test_xlsx_supported(self):
        """Test XLSX extension is supported."""
        config = BulkProcessingConfig()
        assert ".xlsx" in config.supported_extensions

    def test_txt_supported(self):
        """Test TXT extension is supported."""
        config = BulkProcessingConfig()
        assert ".txt" in config.supported_extensions

    def test_png_supported(self):
        """Test PNG extension is supported."""
        config = BulkProcessingConfig()
        assert ".png" in config.supported_extensions

    def test_jpg_supported(self):
        """Test JPG extension is supported."""
        config = BulkProcessingConfig()
        assert ".jpg" in config.supported_extensions


class TestBulkProcessingConfigFromEnv:
    """Tests for loading config from environment variables."""

    def test_from_env_max_documents(self, monkeypatch):
        """Test loading max documents from environment."""
        monkeypatch.setenv("BULK_MAX_DOCUMENTS_PER_FOLDER", "50")
        config = BulkProcessingConfig.from_env()
        assert config.max_documents_per_folder == 50

    def test_from_env_concurrent_documents(self, monkeypatch):
        """Test loading concurrent documents from environment."""
        monkeypatch.setenv("BULK_CONCURRENT_DOCUMENTS", "5")
        config = BulkProcessingConfig.from_env()
        assert config.concurrent_documents == 5

    def test_from_env_parse_timeout(self, monkeypatch):
        """Test loading parse timeout from environment."""
        monkeypatch.setenv("BULK_PARSE_TIMEOUT_SECONDS", "600")
        config = BulkProcessingConfig.from_env()
        assert config.parse_timeout_seconds == 600

    def test_from_env_webhook_enabled(self, monkeypatch):
        """Test loading webhook enabled from environment."""
        monkeypatch.setenv("BULK_WEBHOOK_ENABLED", "false")
        config = BulkProcessingConfig.from_env()
        assert config.webhook_enabled is False

    def test_from_env_webhook_secret(self, monkeypatch):
        """Test loading webhook secret from environment."""
        monkeypatch.setenv("BULK_WEBHOOK_SECRET", "my-secret-key")
        config = BulkProcessingConfig.from_env()
        assert config.webhook_secret == "my-secret-key"

    def test_from_env_generate_summary(self, monkeypatch):
        """Test loading generate summary from environment."""
        monkeypatch.setenv("BULK_DEFAULT_GENERATE_SUMMARY", "false")
        config = BulkProcessingConfig.from_env()
        assert config.default_generate_summary is False

    def test_from_env_num_faqs(self, monkeypatch):
        """Test loading num FAQs from environment."""
        monkeypatch.setenv("BULK_DEFAULT_NUM_FAQS", "20")
        config = BulkProcessingConfig.from_env()
        assert config.default_num_faqs == 20

    def test_from_env_worker_poll_interval(self, monkeypatch):
        """Test loading worker poll interval from environment."""
        monkeypatch.setenv("BULK_WORKER_POLL_INTERVAL_SECONDS", "2.5")
        config = BulkProcessingConfig.from_env()
        assert config.worker_poll_interval_seconds == 2.5


class TestBulkProcessingConfigCustomValues:
    """Tests for custom config values."""

    def test_custom_max_documents(self):
        """Test setting custom max documents."""
        config = BulkProcessingConfig(max_documents_per_folder=100)
        assert config.max_documents_per_folder == 100

    def test_custom_concurrent_documents(self):
        """Test setting custom concurrent documents."""
        config = BulkProcessingConfig(concurrent_documents=10)
        assert config.concurrent_documents == 10

    def test_custom_webhook_secret(self):
        """Test setting custom webhook secret."""
        config = BulkProcessingConfig(webhook_secret="custom-secret")
        assert config.webhook_secret == "custom-secret"

    def test_custom_supported_extensions(self):
        """Test setting custom supported extensions."""
        extensions = [".pdf", ".txt"]
        config = BulkProcessingConfig(supported_extensions=extensions)
        assert config.supported_extensions == extensions


class TestConfigSingleton:
    """Tests for config singleton pattern."""

    def test_get_config_returns_instance(self):
        """Test get_bulk_config returns a config instance."""
        config = get_bulk_config()
        assert isinstance(config, BulkProcessingConfig)

    def test_get_config_returns_same_instance(self):
        """Test get_bulk_config returns the same instance."""
        config1 = get_bulk_config()
        config2 = get_bulk_config()
        assert config1 is config2

    def test_reset_config_clears_singleton(self):
        """Test reset_bulk_config clears the singleton."""
        config1 = get_bulk_config()
        reset_bulk_config()
        config2 = get_bulk_config()
        # They should be different instances after reset
        assert config1 is not config2

    def test_reset_allows_new_config(self, monkeypatch):
        """Test that reset allows loading new config from env."""
        # Get initial config
        config1 = get_bulk_config()
        initial_max_docs = config1.max_documents_per_folder

        # Reset and set new env var
        reset_bulk_config()
        monkeypatch.setenv("BULK_MAX_DOCUMENTS_PER_FOLDER", "999")

        # Get new config - should have new value
        config2 = get_bulk_config()
        assert config2.max_documents_per_folder == 999
        assert config2.max_documents_per_folder != initial_max_docs
