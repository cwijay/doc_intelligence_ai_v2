"""
Usage tracking and enforcement for AI Backend.

Uses local PostgreSQL-based usage tracking (src.core.usage module).
Provides helper functions for checking limits before processing and logging usage after.
"""

import logging
from typing import Optional

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Try to import local usage module, gracefully degrade if not available
try:
    from src.core.usage import (
        get_quota_checker,
        get_usage_service,
        TokenUsage,
        QuotaStatus,
        UsageSummary,
    )

    USAGE_TRACKING_ENABLED = True
except ImportError:
    logger.warning(
        "Usage tracking module not available, usage tracking disabled. "
        "Ensure src.core.usage module is properly configured."
    )
    get_quota_checker = None
    get_usage_service = None
    TokenUsage = None
    QuotaStatus = None
    UsageSummary = None
    USAGE_TRACKING_ENABLED = False


async def check_token_limit_before_processing(
    org_id: str, estimated_tokens: int = 1000
) -> Optional[dict]:
    """
    Check token limit before processing.

    Raises HTTPException with 402 Payment Required if limit exceeded.

    Args:
        org_id: Organization ID
        estimated_tokens: Estimated tokens for the request (default 1000)

    Returns:
        Token limit result dict if tracking enabled, None otherwise

    Raises:
        HTTPException: 402 if token limit would be exceeded
    """
    if not USAGE_TRACKING_ENABLED or not get_quota_checker:
        logger.debug("Usage tracking disabled, skipping token limit check")
        return None

    try:
        checker = get_quota_checker()
        result = await checker.check_quota(
            org_id=org_id,
            usage_type="tokens",
            estimated_usage=estimated_tokens,
        )

        if not result.allowed:
            logger.warning(
                f"Token limit exceeded for org {org_id}: "
                f"used={result.current_usage}, "
                f"limit={result.limit}, "
                f"percentage={result.percentage_used:.1f}%"
            )

            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "token_limit_exceeded",
                    "message": (
                        f"Monthly token limit exceeded. "
                        f"Used: {result.current_usage:,}, "
                        f"Limit: {result.limit:,}"
                    ),
                    "tokens_used": result.current_usage,
                    "monthly_limit": result.limit,
                    "remaining_tokens": result.limit - result.current_usage,
                    "percentage_used": result.percentage_used,
                    "upgrade_hint": result.upgrade_message or "Upgrade your plan for more tokens",
                    "upgrade": {
                        "tier": result.upgrade_tier,
                        "url": result.upgrade_url or f"/settings/billing?upgrade={result.upgrade_tier}",
                    } if result.upgrade_tier else None,
                },
            )

        # Log warning if approaching limit (>80%)
        if result.percentage_used >= 80:
            logger.info(
                f"Organization {org_id} approaching token limit: "
                f"{result.percentage_used:.1f}% used"
            )

        return {
            "tokens_used": result.current_usage,
            "monthly_limit": result.limit,
            "remaining_tokens": result.limit - result.current_usage,
            "percentage_used": result.percentage_used,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log error but don't block processing on tracking failure
        logger.error(f"Error checking token limit: {e}")
        return None


async def check_resource_limit_before_processing(
    org_id: str, resource_type: str, estimated_usage: int = 1
) -> Optional[dict]:
    """
    Check resource limit (file_search_queries, storage_bytes, llamaparse_pages) before processing.

    Raises HTTPException with 402 Payment Required if limit exceeded.

    Args:
        org_id: Organization ID
        resource_type: Resource type (file_search_queries, storage_bytes, llamaparse_pages)
        estimated_usage: Estimated resource usage (default 1)

    Returns:
        Resource limit result dict if tracking enabled, None otherwise

    Raises:
        HTTPException: 402 if resource limit would be exceeded
    """
    if not USAGE_TRACKING_ENABLED or not get_quota_checker:
        logger.debug(f"Usage tracking disabled, skipping {resource_type} limit check")
        return None

    try:
        checker = get_quota_checker()
        result = await checker.check_quota(
            org_id=org_id,
            usage_type=resource_type,
            estimated_usage=estimated_usage,
        )

        if not result.allowed:
            logger.warning(
                f"{resource_type} limit exceeded for org {org_id}: "
                f"used={result.current_usage}, "
                f"limit={result.limit}, "
                f"percentage={result.percentage_used:.1f}%"
            )

            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": f"{resource_type}_limit_exceeded",
                    "message": (
                        f"Monthly {resource_type.replace('_', ' ')} limit exceeded. "
                        f"Used: {result.current_usage:,}, "
                        f"Limit: {result.limit:,}"
                    ),
                    "used": result.current_usage,
                    "limit": result.limit,
                    "remaining": result.limit - result.current_usage,
                    "percentage_used": result.percentage_used,
                    "upgrade_hint": result.upgrade_message or "Upgrade your plan for more resources",
                },
            )

        return {
            "used": result.current_usage,
            "limit": result.limit,
            "remaining": result.limit - result.current_usage,
            "percentage_used": result.percentage_used,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking {resource_type} limit: {e}")
        return None


def log_resource_usage_async(
    org_id: str,
    resource_type: str,
    amount: int,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    extra_data: Optional[dict] = None,
) -> None:
    """
    Log resource usage asynchronously via background queue.

    This enqueues the usage event for non-blocking processing.

    Args:
        org_id: Organization ID
        resource_type: Resource type (file_search_queries, storage_bytes, llamaparse_pages)
        amount: Amount of resource used
        user_id: User ID (optional)
        request_id: Unique request ID for deduplication
        extra_data: Additional metadata
    """
    if not USAGE_TRACKING_ENABLED:
        logger.debug(
            f"Usage tracking disabled, not logging: "
            f"resource={resource_type}, amount={amount}"
        )
        return

    try:
        from src.core.usage import enqueue_resource_usage

        enqueue_resource_usage(
            org_id=org_id,
            resource_type=resource_type,
            amount=amount,
            user_id=user_id,
            request_id=request_id,
            metadata=extra_data,
        )

        logger.debug(
            f"Enqueued resource usage: org={org_id}, type={resource_type}, amount={amount}"
        )
    except Exception as e:
        logger.warning(f"Failed to enqueue resource usage: {e}")


def log_token_usage_async(
    org_id: str,
    user_id: Optional[str],
    feature: str,
    model: str,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    input_cost_usd: float = 0.0,
    output_cost_usd: float = 0.0,
    cached_tokens: int = 0,
    request_id: Optional[str] = None,
    extra_data: Optional[dict] = None,
) -> None:
    """
    Log token usage asynchronously via background queue.

    This enqueues the usage event for non-blocking processing.

    Args:
        org_id: Organization ID
        user_id: User ID (optional)
        feature: Feature name (e.g., "document_agent", "sheets_agent")
        model: Model name (e.g., "gemini-2.5-flash", "gpt-5-mini")
        provider: Provider name (e.g., "google", "openai", "anthropic")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_cost_usd: Cost of input tokens in USD (unused, kept for API compatibility)
        output_cost_usd: Cost of output tokens in USD (unused, kept for API compatibility)
        cached_tokens: Number of cached tokens
        request_id: Unique request ID for deduplication
        extra_data: Additional metadata
    """
    if not USAGE_TRACKING_ENABLED:
        logger.debug(
            f"Usage tracking disabled, not logging: "
            f"feature={feature}, tokens={input_tokens}+{output_tokens}"
        )
        return

    try:
        from src.core.usage import enqueue_token_usage

        enqueue_token_usage(
            org_id=org_id,
            feature=feature,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            user_id=user_id,
            request_id=request_id,
            metadata=extra_data,
        )

        logger.debug(
            f"Enqueued token usage: org={org_id}, feature={feature}, "
            f"model={model}, tokens={input_tokens}+{output_tokens}"
        )
    except Exception as e:
        logger.warning(f"Failed to enqueue token usage: {e}")


async def get_token_usage_summary(org_id: str) -> Optional[dict]:
    """
    Get token usage summary for an organization.

    Args:
        org_id: Organization ID

    Returns:
        Token usage summary dict, or None if tracking disabled
    """
    if not USAGE_TRACKING_ENABLED or not get_usage_service:
        return None

    try:
        service = get_usage_service()
        summary = await service.get_usage_summary(org_id)

        if summary:
            return {
                "organization_id": org_id,
                "tokens_used_this_period": summary.tokens_used,
                "monthly_limit": summary.tokens_limit,
                "remaining_tokens": summary.tokens_limit - summary.tokens_used,
                "percentage_used": summary.tokens_percentage,
                "llamaparse_pages_used": summary.llamaparse_pages_used,
                "llamaparse_pages_limit": summary.llamaparse_pages_limit,
                "file_search_queries_used": summary.file_search_queries_used,
                "file_search_queries_limit": summary.file_search_queries_limit,
                "storage_used_bytes": summary.storage_used_bytes,
                "storage_limit_bytes": summary.storage_limit_bytes,
                "tier": summary.tier,
                "period_start": summary.period_start.isoformat() if summary.period_start else None,
                "period_end": summary.period_end.isoformat() if summary.period_end else None,
            }
        return None
    except Exception as e:
        logger.error(f"Error getting token usage summary: {e}")
        return None


# Re-export decorators for convenience
try:
    from src.core.usage import check_quota, track_resource
except ImportError:
    check_quota = None
    track_resource = None


__all__ = [
    "check_token_limit_before_processing",
    "check_resource_limit_before_processing",
    "log_token_usage_async",
    "log_resource_usage_async",
    "get_token_usage_summary",
    "check_quota",
    "track_resource",
    "USAGE_TRACKING_ENABLED",
]
