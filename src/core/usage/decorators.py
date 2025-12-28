"""
Decorators for quota checking and usage tracking.

Provides:
- @check_quota: Pre-processing quota enforcement (raises HTTP 402 if exceeded)
- @track_resource: Post-processing resource usage logging
"""

import asyncio
import functools
import logging
from typing import Callable, Optional, Any, Union

from fastapi import HTTPException, status

from .quota_checker import get_quota_checker
from .schemas import QuotaStatus

logger = logging.getLogger(__name__)


def check_quota(
    usage_type: str = "tokens",
    estimated_usage: int = 1000,
    org_id_param: str = "org_id",
):
    """
    Decorator to check quota before function execution.

    Raises HTTP 402 (Payment Required) with upgrade CTA if quota exceeded.

    Args:
        usage_type: 'tokens', 'llamaparse_pages', 'file_search_queries', 'storage_bytes'
        estimated_usage: Estimated usage for this request
        org_id_param: Name of parameter containing org_id (in kwargs or function signature)

    Usage:
        @check_quota(usage_type="tokens", estimated_usage=2000)
        async def process_document(request, org_id: str = Depends(get_org_id)):
            ...

        @check_quota(usage_type="llamaparse_pages", estimated_usage=1)
        async def parse_document(request, org_id: str = Depends(get_org_id)):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract org_id from kwargs
            org_id = kwargs.get(org_id_param)

            if not org_id:
                # Try to find in args based on function signature
                import inspect

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if org_id_param in params:
                    idx = params.index(org_id_param)
                    if idx < len(args):
                        org_id = args[idx]

            if not org_id:
                logger.warning(f"Could not extract {org_id_param} from function call, skipping quota check")
                return await func(*args, **kwargs)

            # Check quota
            checker = get_quota_checker()
            try:
                status_result = await checker.check_quota(
                    org_id=org_id,
                    usage_type=usage_type,
                    estimated_usage=estimated_usage,
                )

                if not status_result.allowed:
                    raise HTTPException(
                        status_code=status.HTTP_402_PAYMENT_REQUIRED,
                        detail=_build_quota_exceeded_response(status_result),
                    )

                # Log warning if approaching limit (>80%)
                if status_result.percentage_used >= 80:
                    logger.warning(
                        f"Organization {org_id} approaching {usage_type} limit: "
                        f"{status_result.percentage_used:.1f}% used "
                        f"({status_result.current_usage:,}/{status_result.limit:,})"
                    )

            except HTTPException:
                raise
            except Exception as e:
                # Log error but don't block request if quota check fails
                logger.error(f"Quota check failed for org {org_id}: {e}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def track_resource(
    resource_type: str,
    amount: Optional[int] = None,
    amount_attr: Optional[str] = None,
    org_id_param: str = "org_id",
    user_id_param: str = "user_id",
):
    """
    Decorator to track non-token resource usage after function execution.

    Logs usage asynchronously (fire-and-forget) after function returns.

    Args:
        resource_type: 'llamaparse_pages', 'file_search_queries', 'storage_bytes'
        amount: Fixed amount to log (use if known ahead of time)
        amount_attr: Attribute name in response to get amount from
        org_id_param: Name of parameter containing org_id
        user_id_param: Name of parameter containing user_id

    Usage:
        # Fixed amount
        @track_resource(resource_type="file_search_queries", amount=1)
        async def query_store(request, org_id: str = Depends(get_org_id)):
            ...

        # Dynamic amount from response
        @track_resource(resource_type="llamaparse_pages", amount_attr="page_count")
        async def parse_document(request, org_id: str = Depends(get_org_id)):
            result = await llama_parse(...)
            return {"page_count": result.page_count, ...}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute the function first
            result = await func(*args, **kwargs)

            # Extract org_id
            org_id = kwargs.get(org_id_param)
            user_id = kwargs.get(user_id_param)

            if not org_id:
                import inspect

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                if org_id_param in params:
                    idx = params.index(org_id_param)
                    if idx < len(args):
                        org_id = args[idx]

            if not org_id:
                logger.debug(f"Could not extract {org_id_param}, skipping resource tracking")
                return result

            # Determine amount
            usage_amount = 0
            if amount is not None:
                usage_amount = amount
            elif amount_attr:
                # Try to get from result
                if isinstance(result, dict):
                    usage_amount = result.get(amount_attr, 0)
                elif hasattr(result, amount_attr):
                    usage_amount = getattr(result, amount_attr, 0)
                elif hasattr(result, "__dict__"):
                    usage_amount = getattr(result, amount_attr, 0)
            else:
                usage_amount = 1  # Default to 1 if not specified

            # Log usage asynchronously
            if usage_amount > 0:
                try:
                    asyncio.create_task(
                        _log_resource_async(
                            org_id=org_id,
                            resource_type=resource_type,
                            amount=usage_amount,
                            user_id=user_id,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to schedule resource logging: {e}")

            return result

        return wrapper

    return decorator


async def _log_resource_async(
    org_id: str,
    resource_type: str,
    amount: int,
    user_id: Optional[str] = None,
) -> None:
    """Log resource usage asynchronously."""
    try:
        from .service import get_usage_service

        service = get_usage_service()
        await service.log_resource_usage(
            org_id=org_id,
            resource_type=resource_type,
            amount=amount,
            user_id=user_id,
        )
    except Exception as e:
        logger.warning(f"Resource logging failed: {e}")


def _build_quota_exceeded_response(status: QuotaStatus) -> dict:
    """Build HTTP 402 response body for quota exceeded."""
    usage_label = status.usage_type.replace("_", " ").title()

    response = {
        "error": "quota_exceeded",
        "usage_type": status.usage_type,
        "current_usage": status.current_usage,
        "limit": status.limit,
        "percentage_used": status.percentage_used,
        "message": f"{usage_label} quota exceeded. Used: {status.current_usage:,} / {status.limit:,}",
    }

    if status.upgrade_tier:
        response["upgrade"] = {
            "tier": status.upgrade_tier,
            "message": status.upgrade_message or f"Upgrade to {status.upgrade_tier.title()} for higher limits",
            "cta": "Upgrade Now",
            "url": status.upgrade_url or f"/settings/billing?upgrade={status.upgrade_tier}",
        }
    else:
        response["upgrade"] = {
            "message": "Contact sales for Enterprise+ options",
            "cta": "Contact Sales",
            "url": "/contact-sales",
        }

    return response


__all__ = [
    "check_quota",
    "track_resource",
]
