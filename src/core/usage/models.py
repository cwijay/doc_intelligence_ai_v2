"""
SQLAlchemy models for usage tracking and quota management.

DEPRECATED: This file re-exports models from biz2bricks_core for backwards compatibility.
All model definitions are now in biz2bricks_core.models.usage.

Tables:
- subscription_tiers: Admin-editable tier configuration
- organization_subscriptions: Per-org subscription state and usage counters
- token_usage_records: Granular token usage logs for analytics
- resource_usage_records: Non-token resource tracking (LlamaParse, file search)
- usage_aggregations: Pre-computed rollups for dashboards
"""

# Re-export all models from biz2bricks_core
from biz2bricks_core import (
    SubscriptionTierModel,
    OrganizationSubscriptionModel,
    TokenUsageRecordModel,
    ResourceUsageRecordModel,
    UsageAggregationModel,
    # Aliases
    SubscriptionTier,
    OrganizationSubscription,
    TokenUsageRecord,
    ResourceUsageRecord,
    UsageAggregation,
)

__all__ = [
    "SubscriptionTierModel",
    "OrganizationSubscriptionModel",
    "TokenUsageRecordModel",
    "ResourceUsageRecordModel",
    "UsageAggregationModel",
    # Aliases
    "SubscriptionTier",
    "OrganizationSubscription",
    "TokenUsageRecord",
    "ResourceUsageRecord",
    "UsageAggregation",
]
