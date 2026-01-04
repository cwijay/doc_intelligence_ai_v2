"""
UsageAggregator - Handles usage reporting and history.

Single responsibility: Aggregating and reporting usage data.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, List

from sqlalchemy import text

from .schemas import UsageSummary, UsageBreakdown

logger = logging.getLogger(__name__)


class UsageAggregator:
    """
    Aggregates and reports usage data.

    Responsibilities:
    - Generate usage summaries
    - Retrieve historical usage data
    - Calculate usage breakdowns
    """

    def __init__(self, subscription_manager=None):
        """
        Initialize UsageAggregator.

        Args:
            subscription_manager: Optional SubscriptionManager instance.
                                  If not provided, will be created on first use.
        """
        self._subscription_manager = subscription_manager

    @property
    def subscription_manager(self):
        """Lazy-load subscription manager."""
        if self._subscription_manager is None:
            from .subscription_manager import get_subscription_manager
            self._subscription_manager = get_subscription_manager()
        return self._subscription_manager

    async def get_usage_summary(self, org_id: str) -> Optional[UsageSummary]:
        """
        Get current period usage summary for organization.

        Args:
            org_id: Organization ID

        Returns:
            UsageSummary or None
        """
        subscription = await self.subscription_manager.get_subscription_from_db(org_id)
        if not subscription:
            return None

        return UsageSummary(
            organization_id=org_id,
            tier_id=subscription["tier_id"],
            tier_name=subscription["tier_name"],
            billing_period_start=subscription["current_period_start"],
            billing_period_end=subscription["current_period_end"],
            tokens=self._create_breakdown(
                used=subscription["tokens_used_this_period"],
                limit=subscription["monthly_token_limit"],
            ),
            llamaparse_pages=self._create_breakdown(
                used=subscription["llamaparse_pages_used"],
                limit=subscription["monthly_llamaparse_pages_limit"],
            ),
            file_search_queries=self._create_breakdown(
                used=subscription["file_search_queries_used"],
                limit=subscription["monthly_file_search_queries_limit"],
            ),
            storage=self._create_breakdown(
                used=subscription["storage_used_bytes"],
                limit=subscription["storage_limit_bytes"],
            ),
        )

    def _create_breakdown(self, used: int, limit: int) -> UsageBreakdown:
        """Create a UsageBreakdown from used and limit values."""
        remaining = max(0, limit - used)
        percentage = round(used / limit * 100, 2) if limit > 0 else 100.0
        return UsageBreakdown(
            used=used,
            limit=limit,
            remaining=remaining,
            percentage_used=percentage,
        )

    async def get_usage_history(
        self,
        org_id: str,
        period_type: str = "daily",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 30,
    ) -> List[Dict]:
        """
        Get historical usage data computed on-demand from granular records.

        Args:
            org_id: Organization ID
            period_type: 'daily' or 'monthly'
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum records to return

        Returns:
            List of aggregation records
        """
        try:
            from src.db.connection import db

            async with db.session() as session:
                if session is None:
                    return []

                # Determine date truncation based on period type
                if period_type == "monthly":
                    date_trunc = "DATE_TRUNC('month', created_at)"
                else:
                    date_trunc = "DATE(created_at)"

                # Build date filter conditions
                date_filter = ""
                params: Dict = {"org_id": org_id, "limit": limit}

                if start_date:
                    date_filter += " AND created_at >= :start_date"
                    params["start_date"] = start_date
                if end_date:
                    date_filter += " AND created_at <= :end_date"
                    params["end_date"] = end_date

                # Query token usage records with on-demand aggregation
                token_query = f"""
                    SELECT
                        {date_trunc} as period_date,
                        COALESCE(SUM(total_tokens), 0) as total_tokens,
                        COALESCE(SUM(total_cost_usd), 0) as total_cost_usd,
                        COUNT(*) as total_requests
                    FROM token_usage_records
                    WHERE organization_id = :org_id
                    {date_filter}
                    GROUP BY {date_trunc}
                """

                # Query resource usage records for llamaparse_pages
                llamaparse_query = f"""
                    SELECT
                        {date_trunc} as period_date,
                        COALESCE(SUM(amount), 0) as llamaparse_pages
                    FROM resource_usage_records
                    WHERE organization_id = :org_id
                    AND resource_type = 'llamaparse_pages'
                    {date_filter}
                    GROUP BY {date_trunc}
                """

                # Query resource usage records for file_search_queries
                file_search_query = f"""
                    SELECT
                        {date_trunc} as period_date,
                        COALESCE(SUM(amount), 0) as file_search_queries
                    FROM resource_usage_records
                    WHERE organization_id = :org_id
                    AND resource_type = 'file_search_queries'
                    {date_filter}
                    GROUP BY {date_trunc}
                """

                # Combined query using CTEs for efficiency
                combined_query = f"""
                    WITH token_agg AS (
                        {token_query}
                    ),
                    llamaparse_agg AS (
                        {llamaparse_query}
                    ),
                    file_search_agg AS (
                        {file_search_query}
                    ),
                    all_dates AS (
                        SELECT period_date FROM token_agg
                        UNION
                        SELECT period_date FROM llamaparse_agg
                        UNION
                        SELECT period_date FROM file_search_agg
                    )
                    SELECT
                        ad.period_date as period_start,
                        COALESCE(t.total_tokens, 0) as total_tokens,
                        COALESCE(t.total_cost_usd, 0) as total_cost_usd,
                        COALESCE(t.total_requests, 0) as total_requests,
                        COALESCE(l.llamaparse_pages, 0) as llamaparse_pages,
                        COALESCE(f.file_search_queries, 0) as file_search_queries
                    FROM all_dates ad
                    LEFT JOIN token_agg t ON ad.period_date = t.period_date
                    LEFT JOIN llamaparse_agg l ON ad.period_date = l.period_date
                    LEFT JOIN file_search_agg f ON ad.period_date = f.period_date
                    ORDER BY ad.period_date DESC
                    LIMIT :limit
                """

                result = await session.execute(text(combined_query), params)
                rows = result.fetchall()

                # Calculate period_end based on period_type
                from datetime import timedelta
                history = []
                for row in rows:
                    period_start = row.period_start
                    if period_type == "monthly":
                        # Last day of the month
                        if period_start.month == 12:
                            period_end = period_start.replace(
                                year=period_start.year + 1, month=1, day=1
                            ) - timedelta(days=1)
                        else:
                            period_end = period_start.replace(
                                month=period_start.month + 1, day=1
                            ) - timedelta(days=1)
                    else:
                        # Same day for daily
                        period_end = period_start

                    history.append({
                        "period_start": period_start,
                        "period_end": period_end,
                        "total_tokens": row.total_tokens,
                        "llamaparse_pages": row.llamaparse_pages,
                        "file_search_queries": row.file_search_queries,
                        "total_cost_usd": float(row.total_cost_usd) if row.total_cost_usd else 0.0,
                        "total_requests": row.total_requests,
                    })

                return history

        except Exception as e:
            logger.error(f"Failed to get usage history for org {org_id}: {e}")
            return []

    async def get_feature_breakdown(
        self,
        org_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Get usage breakdown by feature for the specified period.

        Args:
            org_id: Organization ID
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of dicts with: name, tokens_used, percentage, cost_usd
        """
        try:
            from src.db.connection import db

            async with db.session() as session:
                if session is None:
                    return []

                # Build date filter conditions
                date_filter = ""
                params: Dict = {"org_id": org_id}

                if start_date:
                    date_filter += " AND created_at >= :start_date"
                    params["start_date"] = start_date
                if end_date:
                    date_filter += " AND created_at <= :end_date"
                    params["end_date"] = end_date

                # Query token usage records grouped by feature
                query = f"""
                    SELECT
                        feature as name,
                        COALESCE(SUM(total_tokens), 0) as tokens_used,
                        COALESCE(SUM(total_cost_usd), 0) as cost_usd
                    FROM token_usage_records
                    WHERE organization_id = :org_id
                    {date_filter}
                    GROUP BY feature
                    ORDER BY tokens_used DESC
                """

                result = await session.execute(text(query), params)
                rows = result.fetchall()

                if not rows:
                    return []

                # Calculate total tokens for percentage calculation
                total_tokens = sum(row.tokens_used for row in rows)

                breakdown = []
                for row in rows:
                    percentage = (
                        round(row.tokens_used / total_tokens * 100, 2)
                        if total_tokens > 0
                        else 0.0
                    )
                    breakdown.append({
                        "name": row.name or "unknown",
                        "tokens_used": row.tokens_used,
                        "percentage": percentage,
                        "cost_usd": float(row.cost_usd) if row.cost_usd else 0.0,
                    })

                return breakdown

        except Exception as e:
            logger.error(f"Failed to get feature breakdown for org {org_id}: {e}")
            return []


# Module-level instance for convenience
_usage_aggregator: Optional[UsageAggregator] = None


def get_usage_aggregator() -> UsageAggregator:
    """Get or create UsageAggregator instance."""
    global _usage_aggregator
    if _usage_aggregator is None:
        _usage_aggregator = UsageAggregator()
    return _usage_aggregator
