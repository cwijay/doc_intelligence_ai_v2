"""Usage and Subscription API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
Provides usage statistics, quota status, and subscription management.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..dependencies import get_org_id

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Response Schemas
# =============================================================================


class UsageBreakdownItem(BaseModel):
    """Usage breakdown by feature or model."""
    name: str
    tokens_used: int = 0
    percentage: float = 0.0
    cost_usd: float = 0.0


class UsageSummaryResponse(BaseModel):
    """Current period usage summary."""
    success: bool
    organization_id: Optional[str] = None
    tier: Optional[str] = None
    tier_display_name: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    days_remaining: Optional[int] = None

    # Token usage
    tokens_used: int = 0
    tokens_limit: int = 0
    tokens_percentage: float = 0.0
    tokens_remaining: int = 0

    # LlamaParse pages
    llamaparse_pages_used: int = 0
    llamaparse_pages_limit: int = 0
    llamaparse_pages_percentage: float = 0.0

    # File search queries
    file_search_queries_used: int = 0
    file_search_queries_limit: int = 0
    file_search_queries_percentage: float = 0.0

    # Storage
    storage_used_bytes: int = 0
    storage_limit_bytes: int = 0
    storage_percentage: float = 0.0
    storage_used_gb: float = 0.0
    storage_limit_gb: float = 0.0

    # Breakdown by feature
    feature_breakdown: Optional[List[UsageBreakdownItem]] = None

    error: Optional[str] = None


class SubscriptionResponse(BaseModel):
    """Subscription details."""
    success: bool
    organization_id: Optional[str] = None
    tier: Optional[str] = None
    tier_display_name: Optional[str] = None
    status: Optional[str] = None
    billing_cycle: Optional[str] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    # Limits
    monthly_token_limit: int = 0
    monthly_llamaparse_pages: int = 0
    monthly_file_search_queries: int = 0
    storage_gb_limit: float = 0.0
    max_file_size_mb: int = 0
    max_concurrent_jobs: int = 0
    requests_per_minute: int = 0
    requests_per_day: int = 0

    # Features
    features: Optional[dict] = None

    # Pricing
    monthly_price_usd: float = 0.0
    annual_price_usd: float = 0.0

    error: Optional[str] = None


class QuotaStatusResponse(BaseModel):
    """All quota limits and remaining amounts."""
    success: bool

    tokens: Optional[dict] = None
    llamaparse_pages: Optional[dict] = None
    file_search_queries: Optional[dict] = None
    storage: Optional[dict] = None

    all_within_limits: bool = True
    approaching_limit: List[str] = Field(default_factory=list)
    exceeded: List[str] = Field(default_factory=list)

    error: Optional[str] = None


class UsageHistoryItem(BaseModel):
    """Usage history item for a period."""
    date: datetime
    tokens_used: int = 0
    llamaparse_pages: int = 0
    file_search_queries: int = 0
    cost_usd: float = 0.0
    request_count: int = 0


class UsageHistoryResponse(BaseModel):
    """Historical usage data."""
    success: bool
    period: str  # 7d, 30d, 90d
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    history: List[UsageHistoryItem] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error: Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/summary",
    response_model=UsageSummaryResponse,
    operation_id="getUsageSummary",
    summary="Get current period usage summary",
)
async def get_usage_summary(
    org_id: str = Depends(get_org_id),
):
    """
    Get usage summary for the current billing period.

    Returns:
    - Token usage and limits
    - LlamaParse page usage
    - File search query usage
    - Storage usage
    - Breakdown by feature

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.core.usage import get_usage_service

        service = get_usage_service()
        summary = await service.get_usage_summary(org_id)

        if not summary:
            # Try to create a subscription for new organizations
            await service.create_subscription(org_id, tier="free")
            summary = await service.get_usage_summary(org_id)

        if not summary:
            return UsageSummaryResponse(
                success=False,
                error="No subscription found for organization"
            )

        # Calculate days remaining
        days_remaining = None
        if summary.billing_period_end:
            delta = summary.billing_period_end - datetime.utcnow()
            days_remaining = max(0, delta.days)

        # Fetch feature breakdown for current billing period
        breakdown_data = await service.get_feature_breakdown(
            org_id=org_id,
            start_date=summary.billing_period_start,
            end_date=summary.billing_period_end,
        )

        # Map breakdown data to response model
        feature_breakdown = None
        if breakdown_data:
            feature_breakdown = [
                UsageBreakdownItem(
                    name=item.get("name", "unknown"),
                    tokens_used=item.get("tokens_used", 0),
                    percentage=item.get("percentage", 0.0),
                    cost_usd=item.get("cost_usd", 0.0),
                )
                for item in breakdown_data
            ]

        return UsageSummaryResponse(
            success=True,
            organization_id=org_id,
            tier=summary.tier_name,
            tier_display_name=summary.tier_name.title() if summary.tier_name else None,
            period_start=summary.billing_period_start,
            period_end=summary.billing_period_end,
            days_remaining=days_remaining,
            tokens_used=summary.tokens.used,
            tokens_limit=summary.tokens.limit,
            tokens_percentage=summary.tokens.percentage_used,
            tokens_remaining=summary.tokens.remaining,
            llamaparse_pages_used=summary.llamaparse_pages.used,
            llamaparse_pages_limit=summary.llamaparse_pages.limit,
            llamaparse_pages_percentage=summary.llamaparse_pages.percentage_used,
            file_search_queries_used=summary.file_search_queries.used,
            file_search_queries_limit=summary.file_search_queries.limit,
            file_search_queries_percentage=summary.file_search_queries.percentage_used,
            storage_used_bytes=summary.storage.used,
            storage_limit_bytes=summary.storage.limit,
            storage_percentage=summary.storage.percentage_used,
            storage_used_gb=summary.storage.used / (1024 ** 3),
            storage_limit_gb=summary.storage.limit / (1024 ** 3),
            feature_breakdown=feature_breakdown,
        )

    except Exception as e:
        logger.exception(f"Failed to get usage summary: {e}")
        return UsageSummaryResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/subscription",
    response_model=SubscriptionResponse,
    operation_id="getSubscription",
    summary="Get subscription details",
)
async def get_subscription(
    org_id: str = Depends(get_org_id),
):
    """
    Get subscription details including tier, limits, and features.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.core.usage import get_usage_service

        service = get_usage_service()
        sub = await service.get_subscription_from_db(org_id)

        if not sub:
            # Try to create a subscription for new organizations
            sub = await service.create_subscription(org_id, tier="free")

        if not sub:
            return SubscriptionResponse(
                success=False,
                error="No subscription found for organization"
            )

        return SubscriptionResponse(
            success=True,
            organization_id=org_id,
            tier=sub.get("tier"),
            tier_display_name=sub.get("tier_display_name"),
            status=sub.get("status"),
            billing_cycle=sub.get("billing_cycle"),
            period_start=sub.get("current_period_start"),
            period_end=sub.get("current_period_end"),
            monthly_token_limit=sub.get("monthly_token_limit", 0),
            monthly_llamaparse_pages=sub.get("monthly_llamaparse_pages_limit", 0),
            monthly_file_search_queries=sub.get("monthly_file_search_queries_limit", 0),
            storage_gb_limit=sub.get("storage_limit_bytes", 0) / (1024 ** 3),
            max_file_size_mb=sub.get("max_file_size_mb", 0),
            max_concurrent_jobs=sub.get("max_concurrent_jobs", 0),
            requests_per_minute=sub.get("requests_per_minute", 0),
            requests_per_day=sub.get("requests_per_day", 0),
            features=sub.get("features"),
            monthly_price_usd=float(sub.get("monthly_price_usd", 0)),
            annual_price_usd=float(sub.get("annual_price_usd", 0)),
        )

    except Exception as e:
        logger.exception(f"Failed to get subscription: {e}")
        return SubscriptionResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/limits",
    response_model=QuotaStatusResponse,
    operation_id="getQuotaStatus",
    summary="Get all quota limits and status",
)
async def get_quota_status(
    org_id: str = Depends(get_org_id),
):
    """
    Get all quota limits and remaining amounts.

    Returns status for:
    - Tokens
    - LlamaParse pages
    - File search queries
    - Storage

    Also indicates which limits are being approached (>80%) or exceeded.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        from src.core.usage import get_quota_checker

        checker = get_quota_checker()

        # Check all quotas
        tokens = await checker.check_quota(org_id, "tokens", 0)
        pages = await checker.check_quota(org_id, "llamaparse_pages", 0)
        queries = await checker.check_quota(org_id, "file_search_queries", 0)
        storage = await checker.check_quota(org_id, "storage_bytes", 0)

        approaching = []
        exceeded = []

        def check_status(name: str, status):
            if not status.allowed:
                exceeded.append(name)
            elif status.percentage_used >= 80:
                approaching.append(name)

        check_status("tokens", tokens)
        check_status("llamaparse_pages", pages)
        check_status("file_search_queries", queries)
        check_status("storage", storage)

        return QuotaStatusResponse(
            success=True,
            tokens={
                "used": tokens.current_usage,
                "limit": tokens.limit,
                "remaining": tokens.limit - tokens.current_usage,
                "percentage": tokens.percentage_used,
                "allowed": tokens.allowed,
            },
            llamaparse_pages={
                "used": pages.current_usage,
                "limit": pages.limit,
                "remaining": pages.limit - pages.current_usage,
                "percentage": pages.percentage_used,
                "allowed": pages.allowed,
            },
            file_search_queries={
                "used": queries.current_usage,
                "limit": queries.limit,
                "remaining": queries.limit - queries.current_usage,
                "percentage": queries.percentage_used,
                "allowed": queries.allowed,
            },
            storage={
                "used_bytes": storage.current_usage,
                "limit_bytes": storage.limit,
                "remaining_bytes": storage.limit - storage.current_usage,
                "percentage": storage.percentage_used,
                "allowed": storage.allowed,
                "used_gb": storage.current_usage / (1024 ** 3),
                "limit_gb": storage.limit / (1024 ** 3),
            },
            all_within_limits=len(exceeded) == 0,
            approaching_limit=approaching,
            exceeded=exceeded,
        )

    except Exception as e:
        logger.exception(f"Failed to get quota status: {e}")
        return QuotaStatusResponse(
            success=False,
            error=str(e)
        )


@router.get(
    "/history",
    response_model=UsageHistoryResponse,
    operation_id="getUsageHistory",
    summary="Get historical usage data",
)
async def get_usage_history(
    period: str = Query("7d", description="Period: 7d, 30d, or 90d"),
    org_id: str = Depends(get_org_id),
):
    """
    Get historical usage data for the specified period.

    Available periods:
    - 7d: Last 7 days (daily granularity)
    - 30d: Last 30 days (daily granularity)
    - 90d: Last 90 days (weekly granularity)

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    try:
        # Validate period
        valid_periods = {"7d": 7, "14d": 14, "21d": 21, "28d": 28, "30d": 30, "90d": 90}
        if period not in valid_periods:
            return UsageHistoryResponse(
                success=False,
                period=period,
                error=f"Invalid period. Must be one of: {', '.join(valid_periods.keys())}"
            )

        days = valid_periods[period]
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        from src.core.usage import get_usage_service

        service = get_usage_service()
        history_data = await service.get_usage_history(
            org_id=org_id,
            period_type="daily" if days <= 30 else "monthly",
            start_date=start_date,
            end_date=end_date,
            limit=days,
        )

        history = []
        total_tokens = 0
        total_cost = 0.0

        for item in (history_data or []):
            history.append(UsageHistoryItem(
                date=item.get("period_start"),
                tokens_used=item.get("total_tokens", 0),
                llamaparse_pages=item.get("llamaparse_pages", 0),
                file_search_queries=item.get("file_search_queries", 0),
                cost_usd=float(item.get("total_cost_usd", 0.0)),
                request_count=item.get("total_requests", 0),
            ))
            total_tokens += item.get("total_tokens", 0)
            total_cost += float(item.get("total_cost_usd", 0.0))

        return UsageHistoryResponse(
            success=True,
            period=period,
            start_date=start_date,
            end_date=end_date,
            history=history,
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
        )

    except Exception as e:
        logger.exception(f"Failed to get usage history: {e}")
        return UsageHistoryResponse(
            success=False,
            period=period,
            error=str(e)
        )


@router.get(
    "/breakdown",
    response_model=UsageSummaryResponse,
    operation_id="getUsageBreakdown",
    summary="Get usage breakdown by feature",
)
async def get_usage_breakdown(
    org_id: str = Depends(get_org_id),
):
    """
    Get detailed usage breakdown by feature and model.

    Alias for /summary with focus on breakdown data.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    # Reuse the summary endpoint which includes breakdown
    return await get_usage_summary(org_id)
