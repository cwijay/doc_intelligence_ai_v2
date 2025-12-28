"""
Quota checking with in-memory caching for performance.

Provides QuotaChecker class that caches subscription data
to avoid database lookups on every request.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Tuple

from .schemas import QuotaStatus
from .exceptions import QuotaExceededException

logger = logging.getLogger(__name__)

# Default cache TTL in seconds
DEFAULT_CACHE_TTL = 60


class QuotaChecker:
    """
    Checks quota limits with in-memory caching.

    Uses a TTL-based cache to store subscription data and avoid
    database lookups on every request. Cache is invalidated when
    usage is logged or on TTL expiry.
    """

    def __init__(self, cache_ttl_seconds: int = DEFAULT_CACHE_TTL):
        """
        Initialize QuotaChecker.

        Args:
            cache_ttl_seconds: Cache time-to-live in seconds (default: 60)
        """
        self._cache: Dict[str, Tuple[Dict, float]] = {}  # key -> (data, timestamp)
        self._cache_ttl = cache_ttl_seconds
        self._lock = asyncio.Lock()

    def _cache_key(self, org_id: str) -> str:
        """Generate cache key for organization."""
        return f"subscription:{org_id}"

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache:
            return False
        _, timestamp = self._cache[key]
        return (datetime.utcnow().timestamp() - timestamp) < self._cache_ttl

    async def get_subscription(self, org_id: str) -> Optional[Dict]:
        """
        Get subscription data for organization (cached).

        Args:
            org_id: Organization ID

        Returns:
            Subscription data dict or None if not found
        """
        cache_key = self._cache_key(org_id)

        # Check cache first
        async with self._lock:
            if self._is_cache_valid(cache_key):
                data, _ = self._cache[cache_key]
                return data

        # Cache miss - fetch from database
        from .service import get_usage_service
        service = get_usage_service()
        subscription = await service.get_subscription_from_db(org_id)

        if subscription:
            # Update cache
            async with self._lock:
                self._cache[cache_key] = (subscription, datetime.utcnow().timestamp())

        return subscription

    async def check_quota(
        self,
        org_id: str,
        usage_type: str,
        estimated_usage: int = 0,
    ) -> QuotaStatus:
        """
        Check if organization has quota remaining for usage type.

        Args:
            org_id: Organization ID
            usage_type: 'tokens', 'llamaparse_pages', 'file_search_queries', 'storage_bytes'
            estimated_usage: Estimated usage for this request (for pre-check)

        Returns:
            QuotaStatus with allowed flag and usage details
        """
        subscription = await self.get_subscription(org_id)

        if not subscription:
            # No subscription - create one with Free tier
            from .service import get_usage_service
            service = get_usage_service()
            subscription = await service.create_subscription(org_id, "free")

            if not subscription:
                # Still no subscription - allow but log warning
                logger.warning(f"Could not create subscription for org {org_id}, allowing request")
                return QuotaStatus(
                    allowed=True,
                    usage_type=usage_type,
                    current_usage=0,
                    limit=0,
                    remaining=0,
                    percentage_used=0,
                )

        # Get current usage and limit based on type
        usage_map = {
            "tokens": ("tokens_used_this_period", "monthly_token_limit"),
            "llamaparse_pages": ("llamaparse_pages_used", "monthly_llamaparse_pages_limit"),
            "file_search_queries": ("file_search_queries_used", "monthly_file_search_queries_limit"),
            "storage_bytes": ("storage_used_bytes", "storage_limit_bytes"),
        }

        if usage_type not in usage_map:
            raise ValueError(f"Unknown usage type: {usage_type}")

        usage_field, limit_field = usage_map[usage_type]
        current_usage = subscription.get(usage_field, 0)
        limit = subscription.get(limit_field, 0)

        # Calculate status
        remaining = max(0, limit - current_usage)
        percentage = (current_usage / limit * 100) if limit > 0 else 100.0
        allowed = (current_usage + estimated_usage) <= limit

        # Determine upgrade suggestion
        tier_id = subscription.get("tier_id", "free")
        upgrade_tier, upgrade_message, upgrade_url = self._get_upgrade_suggestion(
            tier_id, usage_type
        )

        return QuotaStatus(
            allowed=allowed,
            usage_type=usage_type,
            current_usage=current_usage,
            limit=limit,
            remaining=remaining,
            percentage_used=round(percentage, 2),
            upgrade_tier=upgrade_tier,
            upgrade_message=upgrade_message,
            upgrade_url=upgrade_url,
        )

    async def check_quota_or_raise(
        self,
        org_id: str,
        usage_type: str,
        estimated_usage: int = 0,
    ) -> QuotaStatus:
        """
        Check quota and raise QuotaExceededException if exceeded.

        Args:
            org_id: Organization ID
            usage_type: 'tokens', 'llamaparse_pages', 'file_search_queries', 'storage_bytes'
            estimated_usage: Estimated usage for this request

        Returns:
            QuotaStatus if allowed

        Raises:
            QuotaExceededException if quota exceeded
        """
        status = await self.check_quota(org_id, usage_type, estimated_usage)

        if not status.allowed:
            raise QuotaExceededException(
                usage_type=status.usage_type,
                current_usage=status.current_usage,
                limit=status.limit,
                upgrade_tier=status.upgrade_tier,
                upgrade_message=status.upgrade_message,
                upgrade_url=status.upgrade_url,
            )

        # Log warning if approaching limit (>80%)
        if status.percentage_used >= 80:
            logger.warning(
                f"Organization {org_id} approaching {usage_type} limit: "
                f"{status.percentage_used:.1f}% used ({status.current_usage:,}/{status.limit:,})"
            )

        return status

    def _get_upgrade_suggestion(
        self, current_tier: str, usage_type: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Get upgrade suggestion based on current tier."""
        usage_label = usage_type.replace("_", " ")

        # Handle tier as string or UUID
        tier_name = current_tier.lower() if isinstance(current_tier, str) else "free"

        if "free" in tier_name:
            return (
                "pro",
                f"Upgrade to Pro for 10x more {usage_label}",
                "/settings/billing?upgrade=pro",
            )
        elif "pro" in tier_name:
            return (
                "enterprise",
                f"Upgrade to Enterprise for 10x more {usage_label}",
                "/settings/billing?upgrade=enterprise",
            )

        # Enterprise or unknown - no upgrade suggestion
        return (None, None, None)

    def invalidate_cache(self, org_id: str) -> None:
        """
        Invalidate cache for an organization.

        Call this after updating usage or subscription data.

        Args:
            org_id: Organization ID
        """
        cache_key = self._cache_key(org_id)
        self._cache.pop(cache_key, None)

    def clear_cache(self) -> None:
        """Clear entire cache."""
        self._cache.clear()


# Singleton instance
_quota_checker: Optional[QuotaChecker] = None
_quota_checker_lock = asyncio.Lock()


def get_quota_checker() -> QuotaChecker:
    """
    Get singleton QuotaChecker instance.

    Returns:
        QuotaChecker singleton
    """
    global _quota_checker
    if _quota_checker is None:
        _quota_checker = QuotaChecker()
    return _quota_checker


__all__ = [
    "QuotaChecker",
    "get_quota_checker",
    "DEFAULT_CACHE_TTL",
]
