"""
Usage tracking and enforcement for AI Backend.

Integrates with biz2bricks_core.UsageService for token tracking and limit enforcement.
Provides helper functions for checking limits before processing and logging usage after.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Optional

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Try to import biz2bricks_core, gracefully degrade if not available
try:
    from biz2bricks_core import usage_service, TokenLimitResult

    USAGE_TRACKING_ENABLED = True
except ImportError:
    logger.warning(
        "biz2bricks_core not available, usage tracking disabled. "
        "Install with: pip install biz2bricks-core"
    )
    usage_service = None
    TokenLimitResult = None
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
    if not USAGE_TRACKING_ENABLED or not usage_service:
        logger.debug("Usage tracking disabled, skipping token limit check")
        return None

    try:
        result = await usage_service.check_token_limit(org_id, estimated_tokens)

        if not result.allowed:
            logger.warning(
                f"Token limit exceeded for org {org_id}: "
                f"used={result.tokens_used_this_period}, "
                f"limit={result.monthly_limit}, "
                f"percentage={result.percentage_used:.1f}%"
            )

            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail={
                    "error": "token_limit_exceeded",
                    "message": (
                        f"Monthly token limit exceeded. "
                        f"Used: {result.tokens_used_this_period:,}, "
                        f"Limit: {result.monthly_limit:,}"
                    ),
                    "tokens_used": result.tokens_used_this_period,
                    "monthly_limit": result.monthly_limit,
                    "remaining_tokens": result.remaining_tokens,
                    "percentage_used": result.percentage_used,
                    "upgrade_hint": "Upgrade your plan for more tokens",
                },
            )

        # Log warning if approaching limit (>80%)
        if result.percentage_used >= 80:
            logger.info(
                f"Organization {org_id} approaching token limit: "
                f"{result.percentage_used:.1f}% used"
            )

        return {
            "tokens_used": result.tokens_used_this_period,
            "monthly_limit": result.monthly_limit,
            "remaining_tokens": result.remaining_tokens,
            "percentage_used": result.percentage_used,
        }

    except HTTPException:
        raise
    except Exception as e:
        # Log error but don't block processing on tracking failure
        logger.error(f"Error checking token limit: {e}")
        return None


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
    Log token usage asynchronously (fire-and-forget).

    This should be called after LLM processing completes.
    Uses asyncio.create_task for non-blocking execution.

    Args:
        org_id: Organization ID
        user_id: User ID (optional)
        feature: Feature name (e.g., "document_agent", "sheets_agent")
        model: Model name (e.g., "gemini-2.5-flash", "gpt-4o-mini")
        provider: Provider name (e.g., "google", "openai", "anthropic")
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_cost_usd: Cost of input tokens in USD
        output_cost_usd: Cost of output tokens in USD
        cached_tokens: Number of cached tokens
        request_id: Unique request ID for deduplication
        extra_data: Additional metadata
    """
    if not USAGE_TRACKING_ENABLED or not usage_service:
        logger.debug(
            f"Usage tracking disabled, not logging: "
            f"feature={feature}, tokens={input_tokens}+{output_tokens}"
        )
        return

    async def _log_usage():
        try:
            await usage_service.log_token_usage(
                org_id=org_id,
                user_id=user_id,
                feature=feature,
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=Decimal(str(input_cost_usd)),
                output_cost=Decimal(str(output_cost_usd)),
                cached_tokens=cached_tokens,
                request_id=request_id,
                extra_data=extra_data,
            )

            # Also update tokens used in current period
            total_tokens = input_tokens + output_tokens
            await usage_service.update_tokens_used(org_id, total_tokens)

            logger.debug(
                f"Logged token usage: org={org_id}, feature={feature}, "
                f"model={model}, tokens={input_tokens}+{output_tokens}"
            )
        except Exception as e:
            logger.warning(f"Failed to log token usage: {e}")

    asyncio.create_task(_log_usage())


async def get_token_usage_summary(org_id: str) -> Optional[dict]:
    """
    Get token usage summary for an organization.

    Args:
        org_id: Organization ID

    Returns:
        Token usage summary dict, or None if tracking disabled
    """
    if not USAGE_TRACKING_ENABLED or not usage_service:
        return None

    try:
        result = await usage_service.check_token_limit(org_id, 0)
        return {
            "organization_id": org_id,
            "tokens_used_this_period": result.tokens_used_this_period,
            "monthly_limit": result.monthly_limit,
            "remaining_tokens": result.remaining_tokens,
            "percentage_used": result.percentage_used,
        }
    except Exception as e:
        logger.error(f"Error getting token usage summary: {e}")
        return None
