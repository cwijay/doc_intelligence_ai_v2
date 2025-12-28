"""
Custom exceptions for usage tracking and quota management.
"""

from typing import Optional, Dict, Any


class UsageTrackingError(Exception):
    """Base exception for usage tracking errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class QuotaExceededException(UsageTrackingError):
    """
    Raised when an organization exceeds their quota limits.

    Contains details needed for HTTP 402 response with upgrade CTA.
    """

    def __init__(
        self,
        usage_type: str,
        current_usage: int,
        limit: int,
        upgrade_tier: Optional[str] = None,
        upgrade_message: Optional[str] = None,
        upgrade_url: Optional[str] = None,
    ):
        self.usage_type = usage_type
        self.current_usage = current_usage
        self.limit = limit
        self.upgrade_tier = upgrade_tier
        self.upgrade_message = upgrade_message
        self.upgrade_url = upgrade_url

        usage_label = usage_type.replace("_", " ").title()
        message = f"{usage_label} quota exceeded. Used: {current_usage:,} / {limit:,}"

        super().__init__(
            message=message,
            details={
                "usage_type": usage_type,
                "current_usage": current_usage,
                "limit": limit,
                "upgrade_tier": upgrade_tier,
            },
        )

    def to_response_dict(self) -> Dict[str, Any]:
        """Convert to HTTP 402 response body."""
        response = {
            "error": "quota_exceeded",
            "usage_type": self.usage_type,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "message": self.message,
        }

        if self.upgrade_tier:
            response["upgrade"] = {
                "tier": self.upgrade_tier,
                "message": self.upgrade_message or f"Upgrade to {self.upgrade_tier.title()} for higher limits",
                "url": self.upgrade_url or f"/settings/billing?upgrade={self.upgrade_tier}",
            }

        return response


class SubscriptionNotFoundError(UsageTrackingError):
    """Raised when subscription is not found for an organization."""

    def __init__(self, org_id: str):
        super().__init__(
            message=f"Subscription not found for organization: {org_id}",
            details={"org_id": org_id},
        )


class TierNotFoundError(UsageTrackingError):
    """Raised when subscription tier is not found."""

    def __init__(self, tier_id: str):
        super().__init__(
            message=f"Subscription tier not found: {tier_id}",
            details={"tier_id": tier_id},
        )


class UsageLoggingError(UsageTrackingError):
    """Raised when usage logging fails."""

    def __init__(self, message: str, org_id: Optional[str] = None):
        super().__init__(
            message=message,
            details={"org_id": org_id} if org_id else {},
        )


__all__ = [
    "UsageTrackingError",
    "QuotaExceededException",
    "SubscriptionNotFoundError",
    "TierNotFoundError",
    "UsageLoggingError",
]
