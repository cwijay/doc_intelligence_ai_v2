"""
Usage tracking and quota management module.

Provides token tracking, quota enforcement, and usage analytics
for the Document Intelligence AI system.

Note: SQLAlchemy models are now defined in biz2bricks_core and re-exported here
for backwards compatibility.
"""

from biz2bricks_core import (
    SubscriptionTierModel,
    OrganizationSubscriptionModel,
    TokenUsageRecordModel,
    ResourceUsageRecordModel,
    UsageAggregationModel,
)
from .schemas import (
    TokenUsage,
    QuotaStatus,
    UsageSummary,
    SubscriptionInfo,
)
from .exceptions import (
    QuotaExceededException,
    UsageTrackingError,
)
from .decorators import (
    check_quota,
    track_resource,
    track_tokens,
)
from .callback_handler import TokenTrackingCallbackHandler
from .context import (
    UsageContext,
    usage_context,
    get_current_context,
    set_context,
    clear_context,
)
from .quota_checker import QuotaChecker, get_quota_checker
from .service import UsageTrackingService, get_usage_service
from .usage_queue import (
    UsageQueue,
    UsageEvent,
    get_usage_queue,
    enqueue_token_usage,
    enqueue_resource_usage,
)

__all__ = [
    # Models
    "SubscriptionTierModel",
    "OrganizationSubscriptionModel",
    "TokenUsageRecordModel",
    "ResourceUsageRecordModel",
    "UsageAggregationModel",
    # Schemas
    "TokenUsage",
    "QuotaStatus",
    "UsageSummary",
    "SubscriptionInfo",
    # Exceptions
    "QuotaExceededException",
    "UsageTrackingError",
    # Decorators
    "check_quota",
    "track_resource",
    "track_tokens",
    # Callback Handler
    "TokenTrackingCallbackHandler",
    # Context
    "UsageContext",
    "usage_context",
    "get_current_context",
    "set_context",
    "clear_context",
    # Quota Checker
    "QuotaChecker",
    "get_quota_checker",
    # Service
    "UsageTrackingService",
    "get_usage_service",
    # Queue
    "UsageQueue",
    "UsageEvent",
    "get_usage_queue",
    "enqueue_token_usage",
    "enqueue_resource_usage",
]
