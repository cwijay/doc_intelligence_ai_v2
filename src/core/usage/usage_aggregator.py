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
        Get historical usage data.

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

                query = """
                    SELECT * FROM usage_aggregations
                    WHERE organization_id = :org_id
                    AND period_type = :period_type
                """
                params: Dict = {"org_id": org_id, "period_type": period_type}

                if start_date:
                    query += " AND period_start >= :start_date"
                    params["start_date"] = start_date
                if end_date:
                    query += " AND period_end <= :end_date"
                    params["end_date"] = end_date

                query += " ORDER BY period_start DESC LIMIT :limit"
                params["limit"] = limit

                result = await session.execute(text(query), params)
                rows = result.fetchall()

                return [
                    {
                        "period_start": row.period_start,
                        "period_end": row.period_end,
                        "total_tokens": row.total_tokens,
                        "llamaparse_pages": row.llamaparse_pages,
                        "file_search_queries": row.file_search_queries,
                        "total_cost_usd": row.total_cost_usd,
                        "total_requests": row.total_requests,
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get usage history for org {org_id}: {e}")
            return []


# Module-level instance for convenience
_usage_aggregator: Optional[UsageAggregator] = None


def get_usage_aggregator() -> UsageAggregator:
    """Get or create UsageAggregator instance."""
    global _usage_aggregator
    if _usage_aggregator is None:
        _usage_aggregator = UsageAggregator()
    return _usage_aggregator
