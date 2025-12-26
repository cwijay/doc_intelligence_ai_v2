# Multi-Tenant Token Consumption Tracking & Quota Management System

> **Design Document** for Document Intelligence AI v3.0
> **Status**: Planning
> **Last Updated**: 2025-12-23

---

## Executive Summary

Production-ready system for tracking token consumption, enforcing organization-level quotas, and integrating with Stripe for billing. Inspired by Claude Code's approach: hard limits with upgrade prompts.

**Key Design Principles:**
1. **Token-based limits** - Simple token counting, no complex per-model cost calculations
2. **Database-driven config** - All tier limits, pricing, features stored in DB for admin UI
3. **Decorator-based integration** - Minimal code changes via decorators and callbacks
4. **Organization-level quotas** - Users share org quota pool

---

## Requirements

| Requirement | Decision |
|-------------|----------|
| **Tier Model** | Free + Pro + Enterprise |
| **Limit Scope** | Organization-level (users share org quota) |
| **Quota Exhaustion** | Hard block + 402 response with upgrade CTA |
| **Billing** | Stripe subscriptions + usage-based (Enterprise) |
| **Config Storage** | Database tables (admin UI editable) |

---

## Default Tier Limits

| Tier | Monthly Tokens | LlamaParse Pages | File Search Queries | Storage | Price |
|------|---------------|------------------|---------------------|---------|-------|
| Free | 50,000 | 50 | 100 | 1 GB | $0 |
| Pro | 500,000 | 500 | 1,000 | 10 GB | $29/mo |
| Enterprise | 5,000,000 | 5,000 | 10,000 | 100 GB | $199/mo + overage |

---

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           API Request                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  @check_quota decorator                                              │
│  - Reads org subscription from DB (cached)                          │
│  - Compares usage vs tier limits                                    │
│  - Returns 402 if exceeded                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Agent Execution (LangChain)                                        │
│  - TokenTrackingCallbackHandler attached                            │
│  - Intercepts on_llm_end() for each LLM call                       │
│  - Extracts actual token counts from response                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TokenTrackingService (async, fire-and-forget)                      │
│  - Logs to token_usage_records table                                │
│  - Atomic increment of organization_subscriptions.tokens_used       │
└─────────────────────────────────────────────────────────────────────┘
```

### Schema Design

```
subscription_tiers (admin-managed via UI)
        │
        ▼ (tier_id FK)
organization_subscriptions (1:1 with organizations)
        │
        ▼ (organization_id FK)
token_usage_records (N per org, for analytics)
        │
        ▼ (aggregated into)
usage_aggregations (daily/monthly rollups for dashboards)
```

---

## Database Models

### 1. SubscriptionTierModel (Admin-Editable)

Stores tier configurations. All limits editable via admin UI.

```python
class SubscriptionTierModel(Base):
    __tablename__ = "subscription_tiers"

    id = Column(UUID, primary_key=True)
    tier = Column(String(50), unique=True)  # free, pro, enterprise
    display_name = Column(String(100))
    description = Column(String(500))

    # Limits (primary quota metrics)
    monthly_token_limit = Column(BigInteger, nullable=False)
    monthly_llamaparse_pages = Column(Integer, nullable=False)
    monthly_file_search_queries = Column(Integer, nullable=False)
    storage_gb_limit = Column(Numeric(10, 2), nullable=False)

    # Rate limits
    requests_per_minute = Column(Integer, default=60)
    requests_per_day = Column(Integer, default=10000)
    max_file_size_mb = Column(Integer, default=50)
    max_concurrent_jobs = Column(Integer, default=5)

    # Feature flags (JSONB for flexibility)
    features = Column(JSONB, default=dict)
    # Example: {"rag_enabled": true, "custom_models": false, "priority_support": true}

    # Pricing (for display, Stripe is source of truth)
    monthly_price_usd = Column(Numeric(10, 2), default=0)
    annual_price_usd = Column(Numeric(10, 2), default=0)

    # Stripe integration
    stripe_product_id = Column(String(100))
    stripe_monthly_price_id = Column(String(100))
    stripe_annual_price_id = Column(String(100))

    # Lifecycle
    is_active = Column(Boolean, default=True)  # Soft delete
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

### 2. OrganizationSubscriptionModel

Tracks each org's subscription state and usage counters.

```python
class OrganizationSubscriptionModel(Base):
    __tablename__ = "organization_subscriptions"

    id = Column(UUID, primary_key=True)
    organization_id = Column(UUID, ForeignKey("organizations.id"), unique=True)
    tier_id = Column(UUID, ForeignKey("subscription_tiers.id"))

    # Subscription state
    status = Column(String(50), default="active")  # active, past_due, canceled, trialing
    billing_cycle = Column(String(20), default="monthly")
    current_period_start = Column(DateTime, nullable=False)
    current_period_end = Column(DateTime, nullable=False, index=True)

    # Usage counters (atomic updates via SQL)
    tokens_used_this_period = Column(BigInteger, default=0)
    llamaparse_pages_used = Column(Integer, default=0)
    file_search_queries_used = Column(Integer, default=0)
    storage_used_gb = Column(Numeric(10, 2), default=0)

    # Stripe references
    stripe_customer_id = Column(String(100), unique=True, index=True)
    stripe_subscription_id = Column(String(100), unique=True, index=True)

    # Timestamps
    trial_end = Column(DateTime)
    canceled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

### 3. TokenUsageRecordModel

Granular usage logs for analytics and audit.

```python
class TokenUsageRecordModel(Base):
    __tablename__ = "token_usage_records"

    id = Column(UUID, primary_key=True)
    organization_id = Column(UUID, ForeignKey("organizations.id"), index=True)
    user_id = Column(UUID, ForeignKey("users.id"), index=True)

    # Request identification
    request_id = Column(String(100), unique=True)  # Idempotency key
    session_id = Column(String(100), index=True)

    # Usage details
    feature = Column(String(50), index=True)  # document_agent, sheets_agent, rag
    provider = Column(String(50))  # openai, google, anthropic
    model = Column(String(100))

    # Token counts (actual from API response)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)

    # Metadata
    metadata = Column(JSONB, default=dict)  # document_name, query preview, etc.
    processing_time_ms = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index('idx_usage_org_created', 'organization_id', 'created_at'),
        Index('idx_usage_org_feature', 'organization_id', 'feature', 'created_at'),
    )
```

### 4. UsageAggregationModel

Pre-computed aggregations for dashboard performance.

```python
class UsageAggregationModel(Base):
    __tablename__ = "usage_aggregations"

    id = Column(UUID, primary_key=True)
    organization_id = Column(UUID, ForeignKey("organizations.id"), index=True)

    # Time bucket
    period_type = Column(String(20))  # daily, monthly
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # Aggregated totals
    total_tokens = Column(BigInteger, default=0)
    document_agent_tokens = Column(BigInteger, default=0)
    sheets_agent_tokens = Column(BigInteger, default=0)
    rag_tokens = Column(BigInteger, default=0)

    # Other resources
    llamaparse_pages = Column(Integer, default=0)
    file_search_queries = Column(Integer, default=0)

    # Request stats
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)

    aggregated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('organization_id', 'period_type', 'period_start'),
    )
```

### 5. StripeWebhookEventModel

Idempotency tracking for webhook processing.

```python
class StripeWebhookEventModel(Base):
    __tablename__ = "stripe_webhook_events"

    id = Column(UUID, primary_key=True)
    stripe_event_id = Column(String(100), unique=True, index=True)
    event_type = Column(String(100), index=True)
    organization_id = Column(UUID, index=True)

    processed = Column(Boolean, default=False)
    processed_at = Column(DateTime)
    error_message = Column(String(500))
    payload = Column(JSONB)

    created_at = Column(DateTime, default=datetime.utcnow)
```

### 6. BillingHistoryModel

Audit trail for billing events.

```python
class BillingHistoryModel(Base):
    __tablename__ = "billing_history"

    id = Column(UUID, primary_key=True)
    organization_id = Column(UUID, ForeignKey("organizations.id"), index=True)

    event_type = Column(String(50))  # subscription_created, payment_succeeded, tier_changed

    # Stripe references
    stripe_invoice_id = Column(String(100))
    stripe_payment_intent_id = Column(String(100))

    # Amounts
    amount_usd = Column(Numeric(12, 2))
    currency = Column(String(3), default="USD")

    # Change tracking
    previous_tier = Column(String(50))
    new_tier = Column(String(50))

    description = Column(String(500))
    metadata = Column(JSONB, default=dict)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
```

---

## Core Usage Library

### Module Structure

```
src/core/usage/
    __init__.py              # Public API exports
    service.py               # TokenTrackingService singleton
    extractors.py            # Provider-specific token extraction
    decorators.py            # @track_tokens, @track_resource, @check_quota
    context.py               # UsageContext context manager
    callback_handler.py      # LangChain TokenTrackingCallbackHandler
    quota.py                 # QuotaEnforcer with DB-driven limits
    schemas.py               # TokenUsage, ResourceUsage, QuotaResult
```

### TokenTrackingService

```python
# src/core/usage/service.py

class TokenTrackingService:
    """
    Centralized service for all usage tracking and quota operations.

    Thread-safe singleton with connection pooling and caching.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._quota_cache = TTLCache(maxsize=1000, ttl=60)  # 60s cache
        self._initialized = True

    async def log_usage(
        self,
        org_id: str,
        feature: str,
        usage: TokenUsage,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log token usage (async, non-blocking)."""
        # Insert into token_usage_records
        # Atomic increment organization_subscriptions.tokens_used_this_period
        ...

    async def log_resource(
        self,
        org_id: str,
        resource_type: str,  # llamaparse_pages, file_search_queries
        amount: int,
    ) -> None:
        """Log non-token resource usage."""
        ...

    async def check_quota(
        self,
        org_id: str,
        usage_type: str,
        estimated: int = 0,
    ) -> QuotaResult:
        """
        Check if org has quota available.

        Returns QuotaResult with allowed, current_usage, limit, remaining.
        Uses cached subscription data (60s TTL).
        """
        ...

    async def get_usage_summary(self, org_id: str) -> UsageSummary:
        """Get current period usage summary for dashboard."""
        ...

    async def update_usage_atomic(
        self,
        org_id: str,
        usage_type: str,
        amount: int,
    ) -> None:
        """Atomic increment of usage counter."""
        # Uses SQL: UPDATE ... SET tokens_used = tokens_used + :amount
        ...


def get_usage_service() -> TokenTrackingService:
    """Global accessor for usage service singleton."""
    return TokenTrackingService()
```

### Token Extractors

```python
# src/core/usage/extractors.py

from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class TokenUsage:
    """Standardized token usage from any provider."""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    provider: Optional[str] = None
    model: Optional[str] = None

    def __add__(self, other: 'TokenUsage') -> 'TokenUsage':
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


def extract_usage(response: Any, provider: Optional[str] = None) -> TokenUsage:
    """
    Auto-detect provider and extract actual token counts from response.

    Supports: OpenAI, Gemini, Anthropic
    Falls back to estimation if extraction fails.
    """
    if provider == "openai" or _is_openai_response(response):
        return _extract_openai_usage(response)
    elif provider == "google" or _is_gemini_response(response):
        return _extract_gemini_usage(response)
    elif provider == "anthropic" or _is_anthropic_response(response):
        return _extract_anthropic_usage(response)
    else:
        return _estimate_usage(response)


def _extract_openai_usage(response) -> TokenUsage:
    """Extract from OpenAI response.usage."""
    usage = getattr(response, 'usage', None)
    if not usage:
        return _estimate_usage(response)

    return TokenUsage(
        input_tokens=getattr(usage, 'prompt_tokens', 0),
        output_tokens=getattr(usage, 'completion_tokens', 0),
        total_tokens=getattr(usage, 'total_tokens', 0),
        cached_tokens=getattr(usage, 'prompt_tokens_details', {}).get('cached_tokens', 0),
        provider="openai",
    )


def _extract_gemini_usage(response) -> TokenUsage:
    """Extract from Gemini response.usage_metadata."""
    usage = getattr(response, 'usage_metadata', None)
    if not usage:
        return _estimate_usage(response)

    return TokenUsage(
        input_tokens=getattr(usage, 'prompt_token_count', 0),
        output_tokens=getattr(usage, 'candidates_token_count', 0),
        total_tokens=getattr(usage, 'total_token_count', 0),
        cached_tokens=getattr(usage, 'cached_content_token_count', 0),
        provider="google",
    )


def _extract_anthropic_usage(response) -> TokenUsage:
    """Extract from Anthropic response.usage."""
    usage = getattr(response, 'usage', None)
    if not usage:
        return _estimate_usage(response)

    return TokenUsage(
        input_tokens=getattr(usage, 'input_tokens', 0),
        output_tokens=getattr(usage, 'output_tokens', 0),
        total_tokens=getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0),
        provider="anthropic",
    )


def _estimate_usage(response) -> TokenUsage:
    """Fallback: estimate tokens from content length."""
    content = str(response)
    estimated = int(len(content.split()) * 1.3)
    return TokenUsage(
        input_tokens=0,
        output_tokens=estimated,
        total_tokens=estimated,
    )


def extract_usage_from_langchain_response(response) -> TokenUsage:
    """Extract usage from LangChain LLMResult."""
    # Check response.llm_output for token info
    llm_output = getattr(response, 'llm_output', {}) or {}

    if 'token_usage' in llm_output:
        usage = llm_output['token_usage']
        return TokenUsage(
            input_tokens=usage.get('prompt_tokens', 0),
            output_tokens=usage.get('completion_tokens', 0),
            total_tokens=usage.get('total_tokens', 0),
        )

    # Fallback to estimation
    return _estimate_usage(response)
```

### Decorators

```python
# src/core/usage/decorators.py

import functools
import asyncio
from typing import Optional, Callable
from fastapi import HTTPException, status

from .service import get_usage_service
from .schemas import QuotaResult


def check_quota(
    usage_type: str = "tokens",
    estimated: int = 1000,
    org_id_param: str = "org_id",
):
    """
    Decorator to check quota before function execution.

    Raises HTTP 402 with upgrade CTA if quota exceeded.

    Usage:
        @check_quota(usage_type="tokens", estimated=2000)
        async def process_document(request, org_id: str):
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract org_id from kwargs or function signature
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
                raise ValueError(f"Could not extract {org_id_param} from function call")

            service = get_usage_service()
            result = await service.check_quota(org_id, usage_type, estimated)

            if not result.allowed:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=_build_quota_exceeded_response(result),
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def track_tokens(
    feature: str,
    provider: Optional[str] = None,
    org_id_param: str = "org_id",
):
    """
    Decorator to track token usage from function return value.

    Usage:
        @track_tokens(feature="document_agent", provider="google")
        async def generate_summary(document: str, org_id: str) -> LLMResponse:
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            org_id = kwargs.get(org_id_param)

            result = await func(*args, **kwargs)

            # Extract and log usage (fire-and-forget)
            if org_id:
                from .extractors import extract_usage
                usage = extract_usage(result, provider)
                service = get_usage_service()
                asyncio.create_task(
                    service.log_usage(org_id, feature, usage)
                )

            return result

        return wrapper
    return decorator


def track_resource(
    resource_type: str,
    count: Optional[int] = None,
    count_field: Optional[str] = None,
    org_id_param: str = "org_id",
):
    """
    Decorator to track non-token resource usage.

    Usage:
        @track_resource(resource_type="llamaparse_pages", count_field="page_count")
        async def parse_document(file_path: str, org_id: str) -> ParseResult:
            ...

        @track_resource(resource_type="file_search_queries", count=1)
        async def query_store(prompt: str, org_id: str) -> SearchResult:
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            org_id = kwargs.get(org_id_param)

            result = await func(*args, **kwargs)

            # Determine count
            if count is not None:
                amount = count
            elif count_field and hasattr(result, count_field):
                amount = getattr(result, count_field)
            else:
                amount = 1

            # Log resource usage (fire-and-forget)
            if org_id and amount > 0:
                service = get_usage_service()
                asyncio.create_task(
                    service.log_resource(org_id, resource_type, amount)
                )

            return result

        return wrapper
    return decorator


def _build_quota_exceeded_response(result: QuotaResult) -> dict:
    """Build 402 response with upgrade CTA."""
    response = {
        "error": "quota_exceeded",
        "usage_type": result.usage_type,
        "current_usage": result.current_usage,
        "limit": result.limit,
        "percentage_used": round((result.current_usage / result.limit) * 100, 1),
        "message": f"Your {result.usage_type} quota has been exceeded. "
                   f"Used: {result.current_usage:,} / {result.limit:,}",
    }

    if result.upgrade_tier:
        response["upgrade"] = {
            "tier": result.upgrade_tier,
            "message": f"Upgrade to {result.upgrade_tier.title()} for higher limits",
            "cta": "Upgrade Now",
            "url": f"/settings/billing?upgrade={result.upgrade_tier}",
        }
    else:
        response["upgrade"] = {
            "message": "Contact sales for Enterprise+ options",
            "cta": "Contact Sales",
            "url": "/contact-sales",
        }

    return response
```

### LangChain Callback Handler

```python
# src/core/usage/callback_handler.py

import asyncio
import logging
from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .service import get_usage_service
from .extractors import extract_usage_from_langchain_response, TokenUsage

logger = logging.getLogger(__name__)


class TokenTrackingCallbackHandler(BaseCallbackHandler):
    """
    LangChain callback handler for automatic token tracking.

    Intercepts on_llm_end() to capture usage from response metadata.
    No manual code changes needed - just add to callback list.

    Usage:
        agent = create_react_agent(
            llm,
            tools,
            callbacks=[
                TokenTrackingCallbackHandler(org_id=org_id, feature="document_agent")
            ]
        )
    """

    def __init__(
        self,
        org_id: str,
        feature: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        super().__init__()
        self.org_id = org_id
        self.feature = feature
        self.user_id = user_id
        self.session_id = session_id
        self.accumulated_usage = TokenUsage(0, 0, 0)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs,
    ) -> None:
        """Called when LLM starts generating."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes - extract and log usage."""
        try:
            usage = extract_usage_from_langchain_response(response)
            self.accumulated_usage += usage

            # Fire-and-forget async logging
            service = get_usage_service()

            # Schedule async task
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    service.log_usage(
                        org_id=self.org_id,
                        feature=self.feature,
                        usage=usage,
                        user_id=self.user_id,
                        session_id=self.session_id,
                    )
                )
            except RuntimeError:
                # No running loop - use sync approach or skip
                logger.debug("No running event loop, skipping async usage logging")

        except Exception as e:
            logger.warning(f"Failed to extract/log token usage: {e}")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Called on LLM error."""
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        pass

    def on_tool_end(self, output: str, **kwargs) -> None:
        pass

    @property
    def total_usage(self) -> TokenUsage:
        """Get total accumulated usage across all LLM calls."""
        return self.accumulated_usage

    def reset(self) -> None:
        """Reset accumulated usage counter."""
        self.accumulated_usage = TokenUsage(0, 0, 0)
```

### Context Manager

```python
# src/core/usage/context.py

import asyncio
from typing import Optional
from contextlib import asynccontextmanager

from .callback_handler import TokenTrackingCallbackHandler
from .schemas import TokenUsage
from .service import get_usage_service


class UsageContext:
    """
    Request-scoped usage tracking via context manager.

    Automatically aggregates all LLM calls within the context
    and logs total usage on exit.

    Usage:
        async with UsageContext(org_id, feature="batch_processing") as ctx:
            for doc in documents:
                await process_document(doc)

        print(f"Total tokens: {ctx.total_usage.total_tokens}")
    """

    def __init__(
        self,
        org_id: str,
        feature: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.org_id = org_id
        self.feature = feature
        self.user_id = user_id
        self.session_id = session_id
        self._callback_handler: Optional[TokenTrackingCallbackHandler] = None
        self._total_usage = TokenUsage(0, 0, 0)

    async def __aenter__(self) -> 'UsageContext':
        self._callback_handler = TokenTrackingCallbackHandler(
            org_id=self.org_id,
            feature=self.feature,
            user_id=self.user_id,
            session_id=self.session_id,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._callback_handler:
            self._total_usage = self._callback_handler.total_usage

        # Log final aggregated usage if any
        if self._total_usage.total_tokens > 0:
            service = get_usage_service()
            await service.log_usage(
                org_id=self.org_id,
                feature=self.feature,
                usage=self._total_usage,
                user_id=self.user_id,
                session_id=self.session_id,
            )

    @property
    def callback_handler(self) -> TokenTrackingCallbackHandler:
        """Get callback handler to attach to LangChain agents."""
        if not self._callback_handler:
            raise RuntimeError("UsageContext must be used as async context manager")
        return self._callback_handler

    @property
    def total_usage(self) -> TokenUsage:
        """Get total usage (available after context exit)."""
        return self._total_usage
```

---

## Integration Examples

### Example 1: Agent with Automatic Tracking

```python
from src.core.usage import TokenTrackingCallbackHandler

class DocumentAgent:
    def __init__(self, config: DocumentAgentConfig):
        self.config = config
        self.org_id = None  # Set per-request

    def _init_agent(self, org_id: str):
        self.org_id = org_id

        callbacks = [
            self.middleware.get_callback_handler(),  # Existing
            TokenTrackingCallbackHandler(            # NEW - 1 line
                org_id=org_id,
                feature="document_agent"
            )
        ]

        self.agent = create_react_agent(
            self.llm,
            self.tools,
            callbacks=callbacks
        )
```

### Example 2: API Endpoint with Quota Check

```python
from src.core.usage import check_quota

@router.post("/process")
@check_quota(usage_type="tokens", estimated=2000)  # NEW - 1 line
async def process_document(
    request: DocumentProcessRequest,
    org_id: str = Depends(get_org_id),
):
    # Quota automatically checked before function executes
    # Returns 402 with upgrade CTA if exceeded
    return await agent.process(request)
```

### Example 3: Resource Tracking

```python
from src.core.usage import track_resource

# LlamaParse pages
@track_resource(resource_type="llamaparse_pages", count_field="page_count")
async def parse_document(file_path: str, org_id: str) -> ParseResult:
    result = await llama_parser.parse(file_path)
    return result

# File search queries
@track_resource(resource_type="file_search_queries", count=1)
async def query_store(store, prompt: str, org_id: str) -> SearchResult:
    return await store.query(prompt)
```

### Example 4: Context Manager for Batch Operations

```python
from src.core.usage import UsageContext

async def batch_process_documents(org_id: str, documents: list):
    async with UsageContext(org_id, feature="batch_processing") as ctx:
        results = []
        for doc in documents:
            # All LLM calls tracked automatically
            result = await process_single_document(doc)
            results.append(result)

    # Log shows aggregated usage
    print(f"Batch complete: {ctx.total_usage.total_tokens} tokens used")
    return results
```

---

## API Endpoints

### User-Facing (`/api/v1/billing/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/usage` | GET | Current period usage dashboard |
| `/usage/history` | GET | Historical usage (7d, 30d, 90d) |
| `/subscription` | GET | Subscription details and limits |
| `/subscription/upgrade` | POST | Initiate tier upgrade |
| `/subscription/cancel` | POST | Cancel subscription |
| `/webhooks/stripe` | POST | Stripe webhook handler |

### Admin (`/api/v1/admin/`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tiers` | GET | List all subscription tiers |
| `/tiers` | POST | Create new tier |
| `/tiers/{tier_id}` | GET | Get tier details |
| `/tiers/{tier_id}` | PUT | Update tier limits/features |
| `/tiers/{tier_id}` | DELETE | Soft-delete tier |
| `/organizations` | GET | List orgs with usage stats |
| `/organizations/{org_id}/usage` | GET | Detailed org usage |
| `/organizations/{org_id}/subscription` | PUT | Override subscription |
| `/usage/summary` | GET | Platform-wide usage summary |

---

## 402 Response Format

```json
{
  "error": "quota_exceeded",
  "usage_type": "tokens",
  "current_usage": 50000,
  "limit": 50000,
  "percentage_used": 100.0,
  "message": "Your token quota has been exceeded. Used: 50,000 / 50,000",
  "upgrade": {
    "tier": "pro",
    "message": "Upgrade to Pro for 500,000 tokens/month",
    "cta": "Upgrade Now",
    "url": "/settings/billing?upgrade=pro"
  }
}
```

---

## Files to Create

### Core Usage Library

| File | Purpose |
|------|---------|
| `src/core/usage/__init__.py` | Public API exports |
| `src/core/usage/service.py` | TokenTrackingService singleton |
| `src/core/usage/extractors.py` | Provider-specific token extraction |
| `src/core/usage/decorators.py` | @track_tokens, @track_resource, @check_quota |
| `src/core/usage/context.py` | UsageContext context manager |
| `src/core/usage/callback_handler.py` | TokenTrackingCallbackHandler |
| `src/core/usage/quota.py` | QuotaEnforcer with caching |
| `src/core/usage/schemas.py` | TokenUsage, ResourceUsage, QuotaResult |

### Billing Module

| File | Purpose |
|------|---------|
| `src/billing/__init__.py` | Module exports |
| `src/billing/stripe_service.py` | Stripe API integration |
| `src/billing/subscription_service.py` | Subscription logic |
| `src/billing/webhook_handler.py` | Webhook processing |
| `src/billing/schemas.py` | Pydantic models |

### API Layer

| File | Purpose |
|------|---------|
| `src/api/routers/billing.py` | User billing endpoints |
| `src/api/routers/admin/tiers.py` | Tier CRUD |
| `src/api/routers/admin/organizations.py` | Org management |
| `src/api/routers/admin/usage.py` | Usage analytics |
| `src/api/schemas/billing.py` | Response schemas |
| `src/api/schemas/admin.py` | Admin schemas |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/seed_subscription_tiers.py` | Seed default tiers |
| `scripts/migrate_orgs_to_free.py` | Migrate existing orgs |

---

## Files to Modify (Minimal)

| File | Changes |
|------|---------|
| `biz2bricks_core/models/` | Add 6 new models |
| `src/db/models.py` | Import new models |
| `src/agents/document/core.py` | Add callback handler (1 line) |
| `src/agents/sheets/core.py` | Add callback handler (1 line) |
| `src/api/routers/documents.py` | Add @check_quota decorator |
| `src/api/routers/sheets.py` | Add @check_quota decorator |
| `src/rag/llama_parse_util.py` | Add @track_resource decorator |
| `src/rag/gemini_file_store.py` | Add @track_resource decorator |

**Total: ~10 lines across 8 files**

---

## Environment Variables

```bash
# Stripe
STRIPE_SECRET_KEY=sk_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PUBLISHABLE_KEY=pk_...

# Feature flags
QUOTA_ENFORCEMENT_ENABLED=true
STRIPE_BILLING_ENABLED=true

# Cache settings
QUOTA_CACHE_TTL_SECONDS=60
```

---

## Migration Strategy

1. **Phase 1**: Add database models, run migrations
2. **Phase 2**: Seed default tiers (Free/Pro/Enterprise)
3. **Phase 3**: Create Free subscriptions for existing orgs
4. **Phase 4**: Deploy with `QUOTA_ENFORCEMENT_ENABLED=false`
5. **Phase 5**: Monitor usage logging, verify data accuracy
6. **Phase 6**: Enable enforcement, monitor 402 responses
7. **Phase 7**: Enable Stripe integration
