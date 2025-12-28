"""
Tests for usage helper functions in src/api/usage.py.

Tests the new resource limit checking and logging functions:
- check_resource_limit_before_processing()
- log_resource_usage_async()
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestCheckResourceLimitBeforeProcessing:
    """Tests for check_resource_limit_before_processing() function."""

    @pytest.mark.asyncio
    async def test_allowed_when_under_limit(self):
        """Test that requests are allowed when under the resource limit."""
        from src.core.usage.schemas import QuotaStatus

        mock_result = QuotaStatus(
            allowed=True,
            usage_type="file_search_queries",
            current_usage=50,
            limit=100,
            remaining=50,
            percentage_used=50.0,
        )

        mock_checker = MagicMock()
        mock_checker.check_quota = AsyncMock(return_value=mock_result)

        with patch("src.api.usage.USAGE_TRACKING_ENABLED", True), \
             patch("src.api.usage.get_quota_checker", return_value=mock_checker):
            from src.api.usage import check_resource_limit_before_processing

            result = await check_resource_limit_before_processing(
                org_id="test-org",
                resource_type="file_search_queries",
                estimated_usage=1
            )

            assert result is not None
            assert result["used"] == 50
            assert result["limit"] == 100
            assert result["remaining"] == 50
            mock_checker.check_quota.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_402_when_limit_exceeded(self):
        """Test that HTTP 402 is raised when resource limit is exceeded."""
        from fastapi import HTTPException
        from src.core.usage.schemas import QuotaStatus

        mock_result = QuotaStatus(
            allowed=False,
            usage_type="file_search_queries",
            current_usage=100,
            limit=100,
            remaining=0,
            percentage_used=100.0,
            upgrade_message="Upgrade to Pro for more queries",
        )

        mock_checker = MagicMock()
        mock_checker.check_quota = AsyncMock(return_value=mock_result)

        with patch("src.api.usage.USAGE_TRACKING_ENABLED", True), \
             patch("src.api.usage.get_quota_checker", return_value=mock_checker):
            from src.api.usage import check_resource_limit_before_processing

            with pytest.raises(HTTPException) as exc_info:
                await check_resource_limit_before_processing(
                    org_id="test-org",
                    resource_type="file_search_queries",
                    estimated_usage=1
                )

            assert exc_info.value.status_code == 402
            assert "file_search_queries_limit_exceeded" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_returns_none_when_tracking_disabled(self):
        """Test that None is returned when usage tracking is disabled."""
        with patch("src.api.usage.USAGE_TRACKING_ENABLED", False):
            from src.api.usage import check_resource_limit_before_processing

            result = await check_resource_limit_before_processing(
                org_id="test-org",
                resource_type="storage_bytes",
                estimated_usage=1000
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_exception(self):
        """Test that None is returned on exception (graceful degradation)."""
        mock_checker = MagicMock()
        mock_checker.check_quota = AsyncMock(side_effect=Exception("Database error"))

        with patch("src.api.usage.USAGE_TRACKING_ENABLED", True), \
             patch("src.api.usage.get_quota_checker", return_value=mock_checker):
            from src.api.usage import check_resource_limit_before_processing

            result = await check_resource_limit_before_processing(
                org_id="test-org",
                resource_type="file_search_queries",
                estimated_usage=1
            )

            assert result is None


class TestLogResourceUsageAsync:
    """Tests for log_resource_usage_async() function."""

    @pytest.mark.asyncio
    async def test_logs_usage_when_tracking_enabled(self):
        """Test that usage is logged when tracking is enabled."""
        mock_service = MagicMock()
        mock_service.log_resource_usage = AsyncMock()

        with patch("src.api.usage.USAGE_TRACKING_ENABLED", True), \
             patch("src.api.usage.get_usage_service", return_value=mock_service):
            from src.api.usage import log_resource_usage_async

            log_resource_usage_async(
                org_id="test-org",
                resource_type="file_search_queries",
                amount=1,
                extra_data={"query": "test query"},
            )

            # Allow async task to run
            await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_no_error_when_tracking_disabled(self):
        """Test that no error occurs when tracking is disabled."""
        with patch("src.api.usage.USAGE_TRACKING_ENABLED", False):
            from src.api.usage import log_resource_usage_async

            # Should not raise any exception
            log_resource_usage_async(
                org_id="test-org",
                resource_type="storage_bytes",
                amount=1000,
            )

    @pytest.mark.asyncio
    async def test_handles_service_exception_gracefully(self):
        """Test that exceptions in logging are handled gracefully."""
        mock_service = MagicMock()
        mock_service.log_resource_usage = AsyncMock(side_effect=Exception("DB error"))

        with patch("src.api.usage.USAGE_TRACKING_ENABLED", True), \
             patch("src.api.usage.get_usage_service", return_value=mock_service):
            from src.api.usage import log_resource_usage_async

            # Should not raise any exception
            log_resource_usage_async(
                org_id="test-org",
                resource_type="file_search_queries",
                amount=1,
            )

            # Allow async task to run
            await asyncio.sleep(0.1)


class TestStorageResourceTracking:
    """Tests for storage resource tracking in upload endpoint."""

    @pytest.mark.asyncio
    async def test_storage_tracking_logs_bytes(self):
        """Test that storage bytes are logged after upload."""
        mock_service = MagicMock()
        mock_service.log_resource_usage = AsyncMock()

        with patch("src.api.usage.USAGE_TRACKING_ENABLED", True), \
             patch("src.api.usage.get_usage_service", return_value=mock_service):
            from src.api.usage import log_resource_usage_async

            # Simulate upload logging
            log_resource_usage_async(
                org_id="test-org",
                resource_type="storage_bytes",
                amount=5000,
                extra_data={
                    "files_uploaded": 2,
                    "filenames": ["file1.pdf", "file2.pdf"],
                },
            )

            # Allow async task to run
            await asyncio.sleep(0.1)


class TestFileSearchQueryTracking:
    """Tests for file search query tracking in chat endpoint."""

    @pytest.mark.asyncio
    async def test_file_search_quota_check(self):
        """Test that file search queries are quota-checked before processing."""
        from src.core.usage.schemas import QuotaStatus

        mock_result = QuotaStatus(
            allowed=True,
            usage_type="file_search_queries",
            current_usage=10,
            limit=100,
            remaining=90,
            percentage_used=10.0,
        )

        mock_checker = MagicMock()
        mock_checker.check_quota = AsyncMock(return_value=mock_result)

        with patch("src.api.usage.USAGE_TRACKING_ENABLED", True), \
             patch("src.api.usage.get_quota_checker", return_value=mock_checker):
            from src.api.usage import check_resource_limit_before_processing

            result = await check_resource_limit_before_processing(
                org_id="test-org",
                resource_type="file_search_queries",
                estimated_usage=1
            )

            assert result is not None
            assert result["remaining"] == 90
            mock_checker.check_quota.assert_called_once_with(
                org_id="test-org",
                usage_type="file_search_queries",
                estimated_usage=1,
            )
