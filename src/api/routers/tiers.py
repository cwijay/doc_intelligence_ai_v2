"""Subscription Tiers API endpoints.

Public endpoint (no auth required) for retrieving available subscription tiers.
Used by registration page to display plan options.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter()


# =============================================================================
# Response Schemas
# =============================================================================


class TierFeatures(BaseModel):
    """Feature flags for a tier."""
    document_agent: bool = True
    sheets_agent: bool = True
    rag_search: bool = True
    api_access: bool = True
    custom_models: bool = False
    priority_support: bool = False
    advanced_analytics: bool = False
    team_management: bool = False
    sso: bool = False
    audit_logs: bool = False
    custom_integrations: bool = False
    dedicated_support: bool = False


class TierLimits(BaseModel):
    """Limits for a subscription tier."""
    monthly_tokens: int = Field(..., description="Monthly token limit")
    monthly_tokens_display: str = Field(..., description="Human-readable token limit (e.g., '8M')")
    llamaparse_pages: int = Field(..., description="Monthly LlamaParse page limit")
    file_search_queries: int = Field(..., description="Monthly file search query limit")
    storage_gb: float = Field(..., description="Storage limit in GB")
    requests_per_minute: int = Field(..., description="Rate limit per minute")
    requests_per_day: int = Field(..., description="Rate limit per day")
    max_file_size_mb: int = Field(..., description="Maximum file size in MB")
    max_concurrent_jobs: int = Field(..., description="Maximum concurrent processing jobs")


class TierResponse(BaseModel):
    """Subscription tier information."""
    id: str = Field(..., description="Tier identifier (free, pro, enterprise)")
    name: str = Field(..., description="Display name")
    description: Optional[str] = Field(None, description="Tier description")
    monthly_price_usd: float = Field(..., description="Monthly price in USD")
    annual_price_usd: float = Field(..., description="Annual price in USD")
    limits: TierLimits
    features: TierFeatures
    highlighted: bool = Field(False, description="Whether this tier is highlighted (popular)")
    key_features: List[str] = Field(default_factory=list, description="Key features for display")


class TiersListResponse(BaseModel):
    """Response for listing all available tiers."""
    success: bool
    tiers: List[TierResponse] = Field(default_factory=list)
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================


def _format_token_display(tokens: int) -> str:
    """Format token count for display (e.g., 8000000 -> '8M')."""
    if tokens >= 1_000_000:
        if tokens % 1_000_000 == 0:
            return f"{tokens // 1_000_000}M"
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        if tokens % 1_000 == 0:
            return f"{tokens // 1_000}K"
        return f"{tokens / 1_000:.1f}K"
    return str(tokens)


def _generate_key_features(tier_name: str, limits: TierLimits, features: dict) -> List[str]:
    """Generate key features list for display."""
    key_features = [
        f"{limits.monthly_tokens_display} tokens/month",
        f"{int(limits.storage_gb)} GB storage",
    ]

    if features.get("dedicated_support"):
        key_features.append("Dedicated support")
    elif features.get("priority_support"):
        key_features.append("Priority support")
    else:
        key_features.append("Basic support")

    return key_features


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "",
    response_model=TiersListResponse,
    operation_id="listTiers",
    summary="List available subscription tiers",
)
async def list_tiers():
    """
    Get all available subscription tiers with pricing and limits.

    This is a **public endpoint** - no authentication required.
    Used by the registration page to display plan options.

    Returns:
    - List of active tiers with pricing, limits, and features
    """
    try:
        from src.db.connection import db

        async with db.session() as session:
            if session is None:
                logger.warning("Database disabled, returning empty tiers")
                return TiersListResponse(success=True, tiers=[])

            result = await session.execute(
                text("""
                    SELECT
                        tier,
                        display_name,
                        description,
                        monthly_token_limit,
                        monthly_llamaparse_pages,
                        monthly_file_search_queries,
                        storage_gb_limit,
                        requests_per_minute,
                        requests_per_day,
                        max_file_size_mb,
                        max_concurrent_jobs,
                        features,
                        monthly_price_usd,
                        annual_price_usd,
                        sort_order
                    FROM subscription_tiers
                    WHERE is_active = true
                    ORDER BY sort_order ASC
                """)
            )
            rows = result.fetchall()

            tiers = []
            for row in rows:
                features_dict = row.features or {}

                limits = TierLimits(
                    monthly_tokens=row.monthly_token_limit,
                    monthly_tokens_display=_format_token_display(row.monthly_token_limit),
                    llamaparse_pages=row.monthly_llamaparse_pages,
                    file_search_queries=row.monthly_file_search_queries,
                    storage_gb=float(row.storage_gb_limit),
                    requests_per_minute=row.requests_per_minute,
                    requests_per_day=row.requests_per_day,
                    max_file_size_mb=row.max_file_size_mb,
                    max_concurrent_jobs=row.max_concurrent_jobs,
                )

                features = TierFeatures(
                    document_agent=features_dict.get("document_agent", True),
                    sheets_agent=features_dict.get("sheets_agent", True),
                    rag_search=features_dict.get("rag_search", True),
                    api_access=features_dict.get("api_access", True),
                    custom_models=features_dict.get("custom_models", False),
                    priority_support=features_dict.get("priority_support", False),
                    advanced_analytics=features_dict.get("advanced_analytics", False),
                    team_management=features_dict.get("team_management", False),
                    sso=features_dict.get("sso", False),
                    audit_logs=features_dict.get("audit_logs", False),
                    custom_integrations=features_dict.get("custom_integrations", False),
                    dedicated_support=features_dict.get("dedicated_support", False),
                )

                key_features = _generate_key_features(row.tier, limits, features_dict)

                tiers.append(TierResponse(
                    id=row.tier,
                    name=row.display_name or row.tier.title(),
                    description=row.description,
                    monthly_price_usd=float(row.monthly_price_usd) if row.monthly_price_usd else 0.0,
                    annual_price_usd=float(row.annual_price_usd) if row.annual_price_usd else 0.0,
                    limits=limits,
                    features=features,
                    highlighted=(row.tier == "pro"),  # Pro tier is highlighted
                    key_features=key_features,
                ))

            return TiersListResponse(success=True, tiers=tiers)

    except Exception as e:
        logger.exception(f"Failed to list tiers: {e}")
        return TiersListResponse(
            success=False,
            error=str(e)
        )
