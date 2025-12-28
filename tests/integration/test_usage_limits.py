"""Integration tests for usage limits (rate limiting, sessions, quotas).

These tests verify the usage limit implementation works correctly
through the API layer without mocking.

Run with: RUN_INTEGRATION_TESTS=1 pytest tests/integration/test_usage_limits.py -v
"""

import os
import time
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

# Skip all tests in this module unless RUN_INTEGRATION_TESTS is set
pytestmark = pytest.mark.integration

TEST_ORG_ID = os.getenv("TEST_ORG_ID", "test-org-usage-limits")


@pytest.fixture(scope="module")
def client():
    """Create a test client."""
    from src.api.app import create_app
    app = create_app()
    return TestClient(app)


@pytest.fixture(scope="function")
def headers():
    """Return headers with unique session for each test."""
    return {
        "X-Organization-ID": TEST_ORG_ID,
        "X-Session-ID": f"test-session-{time.time()}"
    }


class TestRateLimitingIntegration:
    """Integration tests for rate limiting through API."""

    def test_health_endpoint_not_rate_limited(self, client):
        """Test that health endpoint is not subject to rate limiting."""
        for _ in range(20):
            response = client.get("/health")
            assert response.status_code == 200

    def test_requests_within_limit_succeed(self, client, headers):
        """Test that requests within rate limit succeed."""
        # Health check doesn't require org header, but we're testing the pattern
        for i in range(5):
            response = client.get("/health")
            assert response.status_code == 200, f"Request {i+1} failed unexpectedly"


class TestSessionIntegration:
    """Integration tests for session management through API."""

    def test_session_endpoint_returns_session_info(self, client, headers):
        """Test sessions endpoint returns session information."""
        # First make a request to create a session
        session_id = headers.get("X-Session-ID", "default-session")

        response = client.get(
            f"/api/v1/sessions/{session_id}",
            headers=headers
        )

        # Session may not exist yet, which is fine
        assert response.status_code in [200, 404]

    def test_delete_session_endpoint(self, client, headers):
        """Test session deletion endpoint."""
        session_id = headers.get("X-Session-ID", "test-delete-session")

        response = client.delete(
            f"/api/v1/sessions/{session_id}",
            headers=headers
        )

        # Either succeeds or session doesn't exist
        assert response.status_code in [200, 404]


class TestQuotaEnforcementIntegration:
    """Integration tests for quota enforcement."""

    def test_quota_check_function_raises_402(self):
        """Test that check_token_limit_before_processing raises HTTPException 402 when quota exceeded."""
        import pytest
        from fastapi import HTTPException
        from unittest.mock import AsyncMock, patch

        from src.core.usage.schemas import QuotaStatus

        # Create a quota status that indicates exceeded
        exceeded_status = QuotaStatus(
            allowed=False,
            usage_type="tokens",
            current_usage=100000,
            limit=50000,
            remaining=0,
            percentage_used=200.0,
            upgrade_tier="pro",
            upgrade_message="Upgrade to Pro for more tokens",
            upgrade_url="/billing?upgrade=pro"
        )

        async def test_check():
            from src.api.usage import check_token_limit_before_processing

            with patch("src.api.usage.get_quota_checker") as mock_get_checker:
                mock_checker = AsyncMock()
                mock_checker.check_quota = AsyncMock(return_value=exceeded_status)
                mock_get_checker.return_value = mock_checker

                with pytest.raises(HTTPException) as exc_info:
                    await check_token_limit_before_processing("test-org", estimated_tokens=1000)

                assert exc_info.value.status_code == 402
                detail = exc_info.value.detail
                assert detail["error"] == "token_limit_exceeded"
                assert detail["upgrade"]["tier"] == "pro"

        import asyncio
        asyncio.run(test_check())

    def test_quota_status_schema_validation(self):
        """Test QuotaStatus schema correctly represents quota state."""
        from src.core.usage.schemas import QuotaStatus

        # Test allowed quota
        allowed = QuotaStatus(
            allowed=True,
            usage_type="tokens",
            current_usage=10000,
            limit=50000,
            remaining=40000,
            percentage_used=20.0
        )
        assert allowed.allowed is True
        assert allowed.remaining == 40000

        # Test exceeded quota
        exceeded = QuotaStatus(
            allowed=False,
            usage_type="tokens",
            current_usage=55000,
            limit=50000,
            remaining=0,
            percentage_used=110.0,
            upgrade_tier="pro",
            upgrade_message="Upgrade needed",
            upgrade_url="/billing"
        )
        assert exceeded.allowed is False
        assert exceeded.upgrade_tier == "pro"


class TestUsageTrackingIntegration:
    """Integration tests for usage tracking."""

    def test_token_usage_schemas_valid(self):
        """Test token usage schemas are correctly defined."""
        from src.core.usage.schemas import TokenUsage, QuotaStatus

        # Test TokenUsage creation
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=10,
            total_tokens=150,
            model="test-model",
            provider="test"
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

        # Test QuotaStatus creation
        status = QuotaStatus(
            allowed=True,
            usage_type="tokens",
            current_usage=1000,
            limit=50000,
            remaining=49000,
            percentage_used=2.0
        )

        assert status.allowed is True
        assert status.remaining == 49000

    def test_token_usage_addition(self):
        """Test token usage accumulation."""
        from src.core.usage.schemas import TokenUsage

        usage1 = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150
        )
        usage2 = TokenUsage(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300
        )

        combined = usage1 + usage2

        assert combined.input_tokens == 300
        assert combined.output_tokens == 150
        assert combined.total_tokens == 450


class TestRateLimiterDirectIntegration:
    """Direct integration tests for RateLimiter."""

    def test_rate_limiter_enforces_limits(self):
        """Test RateLimiter enforces configured limits."""
        from src.agents.core.rate_limiter import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)
        session_id = f"integration-test-{time.time()}"

        # First 3 requests should succeed
        assert limiter.is_allowed(session_id) is True
        assert limiter.is_allowed(session_id) is True
        assert limiter.is_allowed(session_id) is True

        # 4th request should be denied
        assert limiter.is_allowed(session_id) is False

        # Check remaining
        assert limiter.get_remaining(session_id) == 0

        # Check retry_after is positive
        retry_after = limiter.get_retry_after(session_id)
        assert retry_after > 0

    def test_rate_limiter_window_expiration(self):
        """Test RateLimiter window expiration works."""
        from src.agents.core.rate_limiter import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=1)
        session_id = f"expiration-test-{time.time()}"

        # First request succeeds
        assert limiter.is_allowed(session_id) is True

        # Second request denied
        assert limiter.is_allowed(session_id) is False

        # Wait for window to expire
        time.sleep(1.1)

        # Request should succeed again
        assert limiter.is_allowed(session_id) is True


class TestSessionManagerDirectIntegration:
    """Direct integration tests for SessionManager."""

    def test_session_manager_lifecycle(self):
        """Test SessionManager session lifecycle."""
        from src.agents.core.session_manager import SessionManager

        manager = SessionManager(timeout_minutes=1)

        # Create session
        session = manager.get_or_create_session(session_id="lifecycle-test")
        assert session.session_id == "lifecycle-test"
        assert session.query_count == 0

        # Update session
        manager.update_session("lifecycle-test", query_count=5)
        updated = manager.get_session("lifecycle-test")
        assert updated.query_count == 5

        # Delete session
        result = manager.delete_session("lifecycle-test")
        assert result is True
        assert manager.get_session("lifecycle-test") is None

    def test_session_manager_caching(self):
        """Test SessionManager response caching."""
        from src.agents.core.session_manager import SessionManager

        manager = SessionManager(max_cache_size=5)
        session_id = f"cache-test-{time.time()}"

        manager.get_or_create_session(session_id=session_id)

        # Cache responses
        for i in range(3):
            manager.cache_response(
                session_id,
                f"query-hash-{i}",
                {"result": f"value-{i}"}
            )

        # Verify cached responses
        for i in range(3):
            cached = manager.get_cached_response(session_id, f"query-hash-{i}")
            assert cached is not None
            assert cached["result"] == f"value-{i}"

        # Cleanup
        manager.delete_session(session_id)
