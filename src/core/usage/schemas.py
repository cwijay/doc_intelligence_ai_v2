"""
Pydantic schemas for usage tracking and quota management.

Provides data models for token usage, quota status, and API responses.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage data from LLM response."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    provider: Optional[str] = None  # openai, google
    model: Optional[str] = None

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Accumulate token usage."""
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            provider=self.provider or other.provider,
            model=self.model or other.model,
        )


class QuotaStatus(BaseModel):
    """Quota check result for an organization."""
    allowed: bool
    usage_type: str  # tokens, llamaparse_pages, file_search_queries, storage_bytes
    current_usage: int
    limit: int
    remaining: int
    percentage_used: float
    upgrade_tier: Optional[str] = None
    upgrade_message: Optional[str] = None
    upgrade_url: Optional[str] = None


class UsageBreakdown(BaseModel):
    """Usage breakdown for a single resource type."""
    used: int
    limit: int
    remaining: int
    percentage_used: float


class UsageSummary(BaseModel):
    """Current period usage summary for an organization."""
    organization_id: str
    tier_id: str
    tier_name: str
    billing_period_start: datetime
    billing_period_end: datetime
    tokens: UsageBreakdown
    llamaparse_pages: UsageBreakdown
    file_search_queries: UsageBreakdown
    storage: UsageBreakdown  # In bytes


class SubscriptionInfo(BaseModel):
    """Subscription details for an organization."""
    organization_id: str
    tier_id: str
    tier_name: str
    status: str  # active, past_due, canceled, trialing
    billing_cycle: str  # monthly, annual
    billing_period_start: datetime
    billing_period_end: datetime
    limits: Dict[str, int]  # monthly_tokens, monthly_pages, etc.
    features: Dict[str, Any]
    trial_end: Optional[datetime] = None
    created_at: datetime


class TierInfo(BaseModel):
    """Subscription tier information."""
    id: str
    tier: str
    display_name: str
    description: Optional[str] = None
    monthly_token_limit: int
    monthly_llamaparse_pages: int
    monthly_file_search_queries: int
    storage_gb_limit: float
    monthly_price_usd: float
    features: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class DailyUsage(BaseModel):
    """Daily usage data point for history."""
    date: datetime
    tokens: int
    input_tokens: int = 0
    output_tokens: int = 0
    llamaparse_pages: int = 0
    file_search_queries: int = 0
    cost_usd: float = 0.0
    requests: int = 0


class UsageHistory(BaseModel):
    """Historical usage data."""
    organization_id: str
    period_type: str  # daily, monthly
    records: List[DailyUsage]
    total_tokens: int
    total_cost_usd: float


class QuotaExceededResponse(BaseModel):
    """
    Response body for HTTP 402 (Payment Required).

    Returned when organization exceeds quota limits.
    """
    error: str = "quota_exceeded"
    usage_type: str
    current_usage: int
    limit: int
    message: str
    upgrade: Optional[Dict[str, str]] = None  # tier, message, url


class TokenUsageLogRequest(BaseModel):
    """Request to log token usage."""
    org_id: str
    user_id: Optional[str] = None
    feature: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time_ms: Optional[int] = None


class ResourceUsageLogRequest(BaseModel):
    """Request to log resource usage."""
    org_id: str
    user_id: Optional[str] = None
    resource_type: str  # llamaparse_pages, file_search_queries, storage_bytes
    amount: int
    request_id: Optional[str] = None
    file_name: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Model pricing constants (USD per token - divide $/1M by 1,000,000)
MODEL_PRICING = {
    # OpenAI GPT-4o ($2.50/1M input, $10.00/1M output)
    "gpt-4o": {
        "input": Decimal("0.0000025"),
        "output": Decimal("0.00001"),
    },
    "gpt-4o-2024-08-06": {
        "input": Decimal("0.0000025"),
        "output": Decimal("0.00001"),
    },
    # OpenAI GPT-5.1-codex-mini ($0.03/1M input, $0.06/1M output)
    "gpt-5.1-codex-mini": {
        "input": Decimal("0.00000003"),
        "output": Decimal("0.00000006"),
    },
    # OpenAI GPT-5.2 ($0.075/1M input, $0.30/1M output)
    "gpt-5.2": {
        "input": Decimal("0.000000075"),
        "output": Decimal("0.0000003"),
    },
    "gpt-5.2-2025-12-11": {
        "input": Decimal("0.000000075"),
        "output": Decimal("0.0000003"),
    },
    # OpenAI GPT-5-mini ($0.25/1M input, $2.00/1M output)
    "gpt-5-mini": {
        "input": Decimal("0.00000025"),
        "output": Decimal("0.000002"),
    },
    "gpt-5-mini-2025-08-07": {
        "input": Decimal("0.00000025"),
        "output": Decimal("0.000002"),
    },
    # OpenAI GPT-5-nano ($0.05/1M input, $0.40/1M output)
    "gpt-5-nano": {
        "input": Decimal("0.00000005"),
        "output": Decimal("0.0000004"),
    },
    "gpt-5-nano-2025-08-07": {
        "input": Decimal("0.00000005"),
        "output": Decimal("0.0000004"),
    },
    # Google Gemini Flash ($0.075/1M input, $0.30/1M output)
    "gemini-3-flash-preview": {
        "input": Decimal("0.000000075"),
        "output": Decimal("0.0000003"),
    },
    # Google Gemini Pro ($1.25/1M input, $5.00/1M output)
    "gemini-2.5-pro": {
        "input": Decimal("0.00000125"),
        "output": Decimal("0.000005"),
    },
    # Default fallback (use gpt-5-mini pricing)
    "default": {
        "input": Decimal("0.00000025"),
        "output": Decimal("0.000002"),
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[Decimal, Decimal, Decimal]:
    """
    Calculate cost for token usage.

    Returns: (input_cost, output_cost, total_cost)
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    input_cost = Decimal(input_tokens) * pricing["input"]
    output_cost = Decimal(output_tokens) * pricing["output"]
    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost


__all__ = [
    "TokenUsage",
    "QuotaStatus",
    "UsageBreakdown",
    "UsageSummary",
    "SubscriptionInfo",
    "TierInfo",
    "DailyUsage",
    "UsageHistory",
    "QuotaExceededResponse",
    "TokenUsageLogRequest",
    "ResourceUsageLogRequest",
    "MODEL_PRICING",
    "calculate_cost",
]
