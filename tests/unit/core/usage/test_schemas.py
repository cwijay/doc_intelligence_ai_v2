"""Tests for usage tracking schemas."""

import pytest
from decimal import Decimal

from src.core.usage.schemas import (
    TokenUsage,
    QuotaStatus,
    UsageBreakdown,
    calculate_cost,
    MODEL_PRICING,
)


class TestTokenUsage:
    """Tests for TokenUsage model."""

    def test_token_usage_creation(self):
        """Test creating a TokenUsage instance."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=10,
            provider="openai",
            model="gpt-4o",
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150
        assert usage.cached_tokens == 10
        assert usage.provider == "openai"
        assert usage.model == "gpt-4o"

    def test_token_usage_defaults(self):
        """Test default values for TokenUsage."""
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0
        assert usage.cached_tokens == 0
        assert usage.provider is None
        assert usage.model is None

    def test_token_usage_addition(self):
        """Test TokenUsage accumulation via __add__."""
        usage1 = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cached_tokens=10,
            provider="openai",
            model="gpt-4o",
        )
        usage2 = TokenUsage(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            cached_tokens=20,
        )

        combined = usage1 + usage2

        assert combined.input_tokens == 300
        assert combined.output_tokens == 150
        assert combined.total_tokens == 450
        assert combined.cached_tokens == 30
        # Provider/model should come from first non-None
        assert combined.provider == "openai"
        assert combined.model == "gpt-4o"

    def test_token_usage_addition_preserves_provider(self):
        """Test that addition preserves provider from first usage."""
        usage1 = TokenUsage(input_tokens=100, provider="google")
        usage2 = TokenUsage(input_tokens=200, provider="openai")

        combined = usage1 + usage2

        assert combined.provider == "google"


class TestQuotaStatus:
    """Tests for QuotaStatus model."""

    def test_quota_status_allowed(self):
        """Test QuotaStatus when within limits."""
        status = QuotaStatus(
            allowed=True,
            usage_type="tokens",
            current_usage=1000,
            limit=50000,
            remaining=49000,
            percentage_used=2.0,
        )
        assert status.allowed is True
        assert status.current_usage == 1000
        assert status.limit == 50000
        assert status.remaining == 49000
        assert status.percentage_used == 2.0

    def test_quota_status_exceeded(self):
        """Test QuotaStatus when quota exceeded."""
        status = QuotaStatus(
            allowed=False,
            usage_type="tokens",
            current_usage=55000,
            limit=50000,
            remaining=0,
            percentage_used=110.0,
            upgrade_tier="pro",
            upgrade_message="Upgrade to Pro for 500,000 tokens/month",
            upgrade_url="/billing/upgrade",
        )
        assert status.allowed is False
        assert status.upgrade_tier == "pro"
        assert status.upgrade_message is not None


class TestUsageBreakdown:
    """Tests for UsageBreakdown model."""

    def test_usage_breakdown_creation(self):
        """Test creating a UsageBreakdown instance."""
        breakdown = UsageBreakdown(
            used=25000,
            limit=50000,
            remaining=25000,
            percentage_used=50.0,
        )
        assert breakdown.used == 25000
        assert breakdown.limit == 50000
        assert breakdown.remaining == 25000
        assert breakdown.percentage_used == 50.0


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_calculate_cost_known_model_gpt4o(self):
        """Test cost calculation for gpt-4o model."""
        input_cost, output_cost, total_cost = calculate_cost(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )

        # gpt-4o: $2.50 per 1M input, $10.00 per 1M output
        expected_input = Decimal("1000") * Decimal("0.0025")  # 2.5
        expected_output = Decimal("500") * Decimal("0.01")    # 5.0

        assert input_cost == expected_input
        assert output_cost == expected_output
        assert total_cost == expected_input + expected_output

    def test_calculate_cost_known_model_gemini(self):
        """Test cost calculation for gemini-3-flash-preview model."""
        input_cost, output_cost, total_cost = calculate_cost(
            model="gemini-3-flash-preview",
            input_tokens=10000,
            output_tokens=5000,
        )

        # gemini-3-flash-preview: $0.075 per 1M input, $0.30 per 1M output
        expected_input = Decimal("10000") * Decimal("0.000075")
        expected_output = Decimal("5000") * Decimal("0.0003")

        assert input_cost == expected_input
        assert output_cost == expected_output

    def test_calculate_cost_unknown_model_uses_default(self):
        """Test that unknown models use default pricing."""
        input_cost, output_cost, total_cost = calculate_cost(
            model="unknown-model-xyz",
            input_tokens=1000,
            output_tokens=500,
        )

        # Default: $0.10 per 1M input, $0.30 per 1M output
        default_pricing = MODEL_PRICING["default"]
        expected_input = Decimal("1000") * default_pricing["input"]
        expected_output = Decimal("500") * default_pricing["output"]

        assert input_cost == expected_input
        assert output_cost == expected_output

    def test_calculate_cost_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        input_cost, output_cost, total_cost = calculate_cost(
            model="gpt-4o",
            input_tokens=0,
            output_tokens=0,
        )

        assert input_cost == Decimal("0")
        assert output_cost == Decimal("0")
        assert total_cost == Decimal("0")
