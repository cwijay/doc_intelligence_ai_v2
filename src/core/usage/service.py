"""
UsageTrackingService - Facade for usage tracking and quota management.

This service acts as a unified interface to the focused usage components:
- SubscriptionManager: Subscription CRUD operations
- TokenLogger: Token usage logging
- ResourceLogger: Resource usage logging
- UsageAggregator: Reporting and history

The facade maintains backward compatibility with existing code while
delegating to specialized classes for better separation of concerns.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List

from src.core.patterns import ThreadSafeSingleton
from .schemas import TokenUsage, UsageSummary

logger = logging.getLogger(__name__)


class UsageTrackingService(ThreadSafeSingleton):
    """
    Facade for usage tracking and quota management.

    Thread-safe singleton that coordinates:
    - Subscription management
    - Token usage logging
    - Resource usage logging
    - Usage aggregation and reporting

    This class delegates to focused components but maintains the
    existing public API for backward compatibility.
    """

    def _initialize(self) -> None:
        """Initialize component references (lazy-loaded)."""
        self._subscription_manager = None
        self._token_logger = None
        self._resource_logger = None
        self._usage_aggregator = None
        logger.info("UsageTrackingService initialized")

    # =========================================================================
    # Component accessors (lazy loading)
    # =========================================================================

    @property
    def _subscriptions(self):
        """Lazy-load SubscriptionManager."""
        if self._subscription_manager is None:
            from .subscription_manager import get_subscription_manager
            self._subscription_manager = get_subscription_manager()
        return self._subscription_manager

    @property
    def _tokens(self):
        """Lazy-load TokenLogger."""
        if self._token_logger is None:
            from .token_logger import get_token_logger
            self._token_logger = get_token_logger()
        return self._token_logger

    @property
    def _resources(self):
        """Lazy-load ResourceLogger."""
        if self._resource_logger is None:
            from .resource_logger import get_resource_logger
            self._resource_logger = get_resource_logger()
        return self._resource_logger

    @property
    def _aggregator(self):
        """Lazy-load UsageAggregator."""
        if self._usage_aggregator is None:
            from .usage_aggregator import get_usage_aggregator
            self._usage_aggregator = get_usage_aggregator()
        return self._usage_aggregator

    # =========================================================================
    # Subscription Management (delegates to SubscriptionManager)
    # =========================================================================

    async def get_subscription_from_db(self, org_id: str) -> Optional[Dict]:
        """Get subscription data for organization from database."""
        return await self._subscriptions.get_subscription_from_db(org_id)

    async def get_subscription(self, org_id: str) -> Optional[Dict]:
        """Get subscription with automatic billing cycle reset if expired."""
        return await self._subscriptions.get_subscription(org_id)

    async def create_subscription(
        self,
        org_id: str,
        tier: str = "free",
    ) -> Optional[Dict]:
        """Create subscription for organization with specified tier."""
        return await self._subscriptions.create_subscription(org_id, tier)

    async def get_tier(self, tier_id: str) -> Optional[Dict]:
        """Get tier configuration by ID or name."""
        return await self._subscriptions.get_tier(tier_id)

    # =========================================================================
    # Token Usage Logging (delegates to TokenLogger)
    # =========================================================================

    async def log_token_usage(
        self,
        org_id: str,
        feature: str,
        usage: TokenUsage,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        processing_time_ms: Optional[int] = None,
    ) -> None:
        """Log token usage and update organization counters."""
        await self._tokens.log_token_usage(
            org_id=org_id,
            feature=feature,
            usage=usage,
            user_id=user_id,
            request_id=request_id,
            session_id=session_id,
            metadata=metadata,
            processing_time_ms=processing_time_ms,
        )

    # =========================================================================
    # Resource Usage Logging (delegates to ResourceLogger)
    # =========================================================================

    async def log_resource_usage(
        self,
        org_id: str,
        resource_type: str,
        amount: int,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        file_name: Optional[str] = None,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log non-token resource usage."""
        await self._resources.log_resource_usage(
            org_id=org_id,
            resource_type=resource_type,
            amount=amount,
            user_id=user_id,
            request_id=request_id,
            file_name=file_name,
            file_path=file_path,
            metadata=metadata,
        )

    # =========================================================================
    # Usage Summary (delegates to UsageAggregator)
    # =========================================================================

    async def get_usage_summary(self, org_id: str) -> Optional[UsageSummary]:
        """Get current period usage summary for organization."""
        return await self._aggregator.get_usage_summary(org_id)

    async def get_usage_history(
        self,
        org_id: str,
        period_type: str = "daily",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 30,
    ) -> List[Dict]:
        """Get historical usage data."""
        return await self._aggregator.get_usage_history(
            org_id=org_id,
            period_type=period_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

    async def get_feature_breakdown(
        self,
        org_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get usage breakdown by feature."""
        return await self._aggregator.get_feature_breakdown(
            org_id=org_id,
            start_date=start_date,
            end_date=end_date,
        )


# Singleton accessor
def get_usage_service() -> UsageTrackingService:
    """Get singleton UsageTrackingService instance."""
    return UsageTrackingService.get_instance()


__all__ = [
    "UsageTrackingService",
    "get_usage_service",
]
