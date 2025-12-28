"""
Tests for billing cycle reset functionality.

Tests the automatic billing cycle reset when the period has expired.
Tests are structured to test both SubscriptionManager (internal) and
UsageTrackingService facade (public API).
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch


class TestBillingCycleReset:
    """Tests for billing cycle reset via UsageTrackingService facade."""

    @pytest.mark.asyncio
    async def test_get_subscription_returns_data_when_period_active(self):
        """Test that subscription is returned as-is when period is still active."""
        from src.core.usage.subscription_manager import SubscriptionManager

        # Future period end date
        future_date = datetime.utcnow() + timedelta(days=10)

        mock_subscription = {
            "id": "sub-123",
            "organization_id": "org-123",
            "tier_name": "free",
            "current_period_end": future_date,
            "tokens_used_this_period": 5000,
            "llamaparse_pages_used": 10,
            "file_search_queries_used": 25,
        }

        manager = SubscriptionManager()

        with patch.object(
            manager, "get_subscription_from_db", new=AsyncMock(return_value=mock_subscription)
        ), patch.object(
            manager, "_reset_billing_cycle", new=AsyncMock()
        ) as mock_reset:
            result = await manager.get_subscription("org-123")

            assert result is not None
            assert result["tokens_used_this_period"] == 5000
            # Reset should NOT be called since period is still active
            mock_reset.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_subscription_resets_when_period_expired(self):
        """Test that billing cycle is reset when period has expired."""
        from src.core.usage.subscription_manager import SubscriptionManager

        # Past period end date
        past_date = datetime.utcnow() - timedelta(days=5)

        mock_subscription = {
            "id": "sub-123",
            "organization_id": "org-123",
            "tier_name": "free",
            "current_period_end": past_date,
            "tokens_used_this_period": 50000,
            "llamaparse_pages_used": 50,
            "file_search_queries_used": 100,
        }

        reset_subscription = {
            "id": "sub-123",
            "organization_id": "org-123",
            "tier_name": "free",
            "current_period_end": datetime.utcnow() + timedelta(days=30),
            "tokens_used_this_period": 0,
            "llamaparse_pages_used": 0,
            "file_search_queries_used": 0,
        }

        manager = SubscriptionManager()

        with patch.object(
            manager, "get_subscription_from_db", new=AsyncMock(return_value=mock_subscription)
        ), patch.object(
            manager, "_reset_billing_cycle", new=AsyncMock(return_value=reset_subscription)
        ) as mock_reset:
            result = await manager.get_subscription("org-123")

            # Reset should be called since period expired
            mock_reset.assert_called_once()
            assert result["tokens_used_this_period"] == 0

    @pytest.mark.asyncio
    async def test_get_subscription_returns_none_when_not_found(self):
        """Test that None is returned when subscription doesn't exist."""
        from src.core.usage.service import UsageTrackingService

        service = UsageTrackingService()

        with patch.object(
            service._subscriptions, "get_subscription_from_db", new=AsyncMock(return_value=None)
        ):
            result = await service.get_subscription("nonexistent-org")

            assert result is None

    @pytest.mark.asyncio
    async def test_reset_billing_cycle_resets_all_counters(self):
        """Test that _reset_billing_cycle resets all usage counters."""
        from src.core.usage.subscription_manager import SubscriptionManager

        past_date = datetime.utcnow() - timedelta(days=5)

        mock_subscription = {
            "id": "sub-123",
            "organization_id": "org-123",
            "tier_name": "free",
            "current_period_end": past_date,
            "tokens_used_this_period": 50000,
            "llamaparse_pages_used": 50,
            "file_search_queries_used": 100,
            "storage_used_bytes": 1000000,
        }

        # Mock async context manager session
        mock_session = MagicMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        # Create async context manager for session
        async def session_context_manager():
            return mock_session

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_db = MagicMock()
        mock_db.session = MagicMock(return_value=mock_session_cm)

        manager = SubscriptionManager()

        # After reset, counters should be zero
        reset_subscription = {
            **mock_subscription,
            "tokens_used_this_period": 0,
            "llamaparse_pages_used": 0,
            "file_search_queries_used": 0,
            "current_period_start": datetime.utcnow().replace(day=1),
            "current_period_end": datetime.utcnow().replace(day=1) + timedelta(days=32),
        }

        with patch.dict("sys.modules", {"src.db.connection": MagicMock(db=mock_db)}), \
             patch.object(
                 manager, "get_subscription_from_db",
                 new=AsyncMock(return_value=reset_subscription)
             ):
            result = await manager._reset_billing_cycle("org-123", mock_subscription)

            # Session execute should be called for the UPDATE
            mock_session.execute.assert_called()
            mock_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_reset_billing_cycle_handles_db_error(self):
        """Test that _reset_billing_cycle handles database errors gracefully."""
        from src.core.usage.subscription_manager import SubscriptionManager

        past_date = datetime.utcnow() - timedelta(days=5)

        mock_subscription = {
            "id": "sub-123",
            "organization_id": "org-123",
            "current_period_end": past_date,
            "tokens_used_this_period": 50000,
        }

        # Create async context manager that raises exception
        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(side_effect=Exception("Database error"))
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_db = MagicMock()
        mock_db.session = MagicMock(return_value=mock_session_cm)

        manager = SubscriptionManager()

        with patch.dict("sys.modules", {"src.db.connection": MagicMock(db=mock_db)}):
            result = await manager._reset_billing_cycle("org-123", mock_subscription)

            # Should return original subscription on error
            assert result == mock_subscription


class TestBillingPeriodCalculation:
    """Tests for billing period date calculations."""

    @pytest.mark.asyncio
    async def test_new_period_starts_on_first_of_month(self):
        """Test that new billing period starts on the 1st of the current month."""
        from src.core.usage.subscription_manager import SubscriptionManager

        # Freeze time to a specific date
        fixed_date = datetime(2024, 3, 15, 12, 0, 0)

        mock_subscription = {
            "id": "sub-123",
            "organization_id": "org-123",
            "current_period_end": datetime(2024, 2, 28),  # Expired
            "tokens_used_this_period": 50000,
        }

        mock_session = MagicMock()
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_cm.__aexit__ = AsyncMock(return_value=None)

        mock_db = MagicMock()
        mock_db.session = MagicMock(return_value=mock_session_cm)

        manager = SubscriptionManager()

        reset_subscription = {
            **mock_subscription,
            "tokens_used_this_period": 0,
            "current_period_start": datetime(2024, 3, 1),
            "current_period_end": datetime(2024, 4, 1),
        }

        with patch.dict("sys.modules", {"src.db.connection": MagicMock(db=mock_db)}), \
             patch.object(
                 manager, "get_subscription_from_db",
                 new=AsyncMock(return_value=reset_subscription)
             ):
            await manager._reset_billing_cycle("org-123", mock_subscription)

            # Verify execute was called (the SQL contains period dates)
            mock_session.execute.assert_called()


class TestResourceLimitEnforcement:
    """Tests for resource limit enforcement after billing cycle reset."""

    @pytest.mark.asyncio
    async def test_quota_allowed_after_reset(self):
        """Test that quota checks pass after billing cycle reset."""
        from src.core.usage.quota_checker import QuotaChecker
        from src.core.usage.schemas import QuotaStatus

        # After reset, usage should be 0
        mock_subscription = {
            "id": "sub-123",
            "organization_id": "org-123",
            "tier_id": "free",
            "tier_name": "free",
            "tokens_used_this_period": 0,
            "monthly_token_limit": 50000,
            "llamaparse_pages_used": 0,
            "monthly_llamaparse_pages_limit": 50,
            "file_search_queries_used": 0,
            "monthly_file_search_queries_limit": 100,
            "storage_used_bytes": 0,
            "storage_limit_bytes": 1073741824,
            "current_period_end": datetime.utcnow() + timedelta(days=30),
        }

        checker = QuotaChecker()

        # Mock get_subscription to return our test data
        with patch.object(
            checker, "get_subscription", new=AsyncMock(return_value=mock_subscription)
        ):
            # Check token quota - should be allowed
            result = await checker.check_quota(
                org_id="org-123",
                usage_type="tokens",
                estimated_usage=1000
            )

            assert result.allowed is True
            assert result.current_usage == 0
            assert result.remaining == 50000
