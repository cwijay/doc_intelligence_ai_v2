"""
SubscriptionManager - Handles subscription CRUD operations.

Single responsibility: Managing organization subscriptions and tiers.
"""

import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, Any

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Tier cache: maps tier_id -> (tier_data, cached_at_timestamp)
# Tiers rarely change, so 1-hour TTL is appropriate
_tier_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
TIER_CACHE_TTL_SECONDS = 3600  # 1 hour


class SubscriptionManager:
    """
    Manages organization subscriptions and tier configurations.

    Responsibilities:
    - Fetch subscription data
    - Create new subscriptions
    - Reset billing cycles
    - Get tier configurations
    """

    async def get_subscription_from_db(self, org_id: str) -> Optional[Dict]:
        """
        Get subscription data for organization from database.

        Args:
            org_id: Organization ID

        Returns:
            Subscription data dict or None
        """
        try:
            from src.db.connection import db

            async with db.session() as session:
                if session is None:
                    logger.warning("Database disabled, returning None")
                    return None

                result = await session.execute(
                    text("""
                        SELECT
                            os.id,
                            os.organization_id,
                            os.tier_id::text,
                            st.tier as tier_name,
                            os.status,
                            os.billing_cycle,
                            os.current_period_start,
                            os.current_period_end,
                            os.tokens_used_this_period,
                            os.llamaparse_pages_used,
                            os.file_search_queries_used,
                            os.storage_used_bytes,
                            os.monthly_token_limit,
                            os.monthly_llamaparse_pages_limit,
                            os.monthly_file_search_queries_limit,
                            os.storage_limit_bytes,
                            os.stripe_customer_id,
                            os.stripe_subscription_id,
                            os.created_at,
                            os.updated_at
                        FROM organization_subscriptions os
                        JOIN subscription_tiers st ON os.tier_id = st.id
                        WHERE os.organization_id = :org_id
                    """),
                    {"org_id": org_id}
                )
                row = result.fetchone()

                if not row:
                    return None

                return {
                    "id": str(row.id),
                    "organization_id": row.organization_id,
                    "tier_id": row.tier_id,
                    "tier_name": row.tier_name,
                    "status": row.status,
                    "billing_cycle": row.billing_cycle,
                    "current_period_start": row.current_period_start,
                    "current_period_end": row.current_period_end,
                    "tokens_used_this_period": row.tokens_used_this_period,
                    "llamaparse_pages_used": row.llamaparse_pages_used,
                    "file_search_queries_used": row.file_search_queries_used,
                    "storage_used_bytes": row.storage_used_bytes,
                    "monthly_token_limit": row.monthly_token_limit,
                    "monthly_llamaparse_pages_limit": row.monthly_llamaparse_pages_limit,
                    "monthly_file_search_queries_limit": row.monthly_file_search_queries_limit,
                    "storage_limit_bytes": row.storage_limit_bytes,
                    "stripe_customer_id": row.stripe_customer_id,
                    "stripe_subscription_id": row.stripe_subscription_id,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }

        except Exception as e:
            logger.error(f"Failed to get subscription for org {org_id}: {e}")
            return None

    async def get_subscription(self, org_id: str) -> Optional[Dict]:
        """
        Get subscription with automatic billing cycle reset if expired.

        If the current billing period has ended, resets usage counters
        and advances to the new billing period.

        Args:
            org_id: Organization ID

        Returns:
            Subscription data dict or None
        """
        subscription = await self.get_subscription_from_db(org_id)
        if not subscription:
            return None

        # Check if billing period has ended
        period_end = subscription.get("current_period_end")
        if period_end and period_end < datetime.utcnow():
            # Reset billing cycle
            subscription = await self._reset_billing_cycle(org_id, subscription)

        return subscription

    async def _reset_billing_cycle(
        self, org_id: str, subscription: Dict
    ) -> Optional[Dict]:
        """
        Reset billing cycle counters and advance to new period.

        Args:
            org_id: Organization ID
            subscription: Current subscription data

        Returns:
            Updated subscription data
        """
        try:
            from src.db.connection import db

            async with db.session() as session:
                if session is None:
                    logger.warning("Database disabled, cannot reset billing cycle")
                    return subscription

                # Calculate new billing period
                now = datetime.utcnow()
                new_period_start = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                new_period_end = (new_period_start + timedelta(days=32)).replace(day=1)

                # Reset usage counters and advance period
                await session.execute(
                    text("""
                        UPDATE organization_subscriptions
                        SET
                            tokens_used_this_period = 0,
                            llamaparse_pages_used = 0,
                            file_search_queries_used = 0,
                            current_period_start = :period_start,
                            current_period_end = :period_end,
                            updated_at = :now
                        WHERE organization_id = :org_id
                    """),
                    {
                        "org_id": org_id,
                        "period_start": new_period_start,
                        "period_end": new_period_end,
                        "now": now,
                    }
                )
                await session.commit()

                logger.info(
                    f"Reset billing cycle for org {org_id}: "
                    f"new period {new_period_start.date()} to {new_period_end.date()}"
                )

                # Return updated subscription
                return await self.get_subscription_from_db(org_id)

        except Exception as e:
            logger.error(f"Failed to reset billing cycle for org {org_id}: {e}")
            return subscription

    async def create_subscription(
        self,
        org_id: str,
        tier: str = "free",
    ) -> Optional[Dict]:
        """
        Create subscription for organization with specified tier.

        Args:
            org_id: Organization ID
            tier: Tier name ('free', 'pro', 'enterprise')

        Returns:
            Created subscription data dict
        """
        try:
            from src.db.connection import db

            async with db.session() as session:
                if session is None:
                    logger.warning("Database disabled, returning None")
                    return None

                # Get tier configuration
                tier_result = await session.execute(
                    text(
                        "SELECT * FROM subscription_tiers "
                        "WHERE tier = :tier AND is_active = true"
                    ),
                    {"tier": tier}
                )
                tier_row = tier_result.fetchone()

                if not tier_row:
                    logger.error(f"Tier not found: {tier}")
                    return None

                # Calculate billing period
                now = datetime.utcnow()
                period_start = now.replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
                period_end = (period_start + timedelta(days=32)).replace(day=1)

                # Convert storage GB to bytes
                storage_limit_bytes = int(
                    float(tier_row.storage_gb_limit) * 1024 * 1024 * 1024
                )

                # Create subscription
                subscription_id = uuid.uuid4()
                await session.execute(
                    text("""
                        INSERT INTO organization_subscriptions (
                            id, organization_id, tier_id, status, billing_cycle,
                            current_period_start, current_period_end,
                            tokens_used_this_period, llamaparse_pages_used,
                            file_search_queries_used, storage_used_bytes,
                            monthly_token_limit, monthly_llamaparse_pages_limit,
                            monthly_file_search_queries_limit, storage_limit_bytes,
                            created_at, updated_at
                        ) VALUES (
                            :id, :org_id, :tier_id, 'active', 'monthly',
                            :period_start, :period_end,
                            0, 0, 0, 0,
                            :token_limit, :pages_limit, :queries_limit, :storage_limit,
                            :now, :now
                        )
                        ON CONFLICT (organization_id) DO NOTHING
                    """),
                    {
                        "id": subscription_id,
                        "org_id": org_id,
                        "tier_id": tier_row.id,
                        "period_start": period_start,
                        "period_end": period_end,
                        "token_limit": tier_row.monthly_token_limit,
                        "pages_limit": tier_row.monthly_llamaparse_pages,
                        "queries_limit": tier_row.monthly_file_search_queries,
                        "storage_limit": storage_limit_bytes,
                        "now": now,
                    }
                )
                await session.commit()

                logger.info(f"Created {tier} subscription for org {org_id}")

                # Return the created subscription
                return await self.get_subscription_from_db(org_id)

        except Exception as e:
            logger.error(f"Failed to create subscription for org {org_id}: {e}")
            return None

    async def get_tier(self, tier_id: str) -> Optional[Dict]:
        """
        Get tier configuration by ID or name.

        Uses TTL-based cache since tiers rarely change.

        Args:
            tier_id: Tier UUID or name

        Returns:
            Tier data dict or None
        """
        # Check cache first
        if tier_id in _tier_cache:
            cached_data, cached_at = _tier_cache[tier_id]
            if time.time() - cached_at < TIER_CACHE_TTL_SECONDS:
                logger.debug(f"Tier cache hit for {tier_id}")
                return cached_data

        try:
            from src.db.connection import db

            async with db.session() as session:
                if session is None:
                    return None

                # Try by ID first, then by name
                result = await session.execute(
                    text("""
                        SELECT * FROM subscription_tiers
                        WHERE id::text = :tier_id OR tier = :tier_id
                        LIMIT 1
                    """),
                    {"tier_id": tier_id}
                )
                row = result.fetchone()

                if not row:
                    return None

                tier_data = {
                    "id": str(row.id),
                    "tier": row.tier,
                    "display_name": row.display_name,
                    "description": row.description,
                    "monthly_token_limit": row.monthly_token_limit,
                    "monthly_llamaparse_pages": row.monthly_llamaparse_pages,
                    "monthly_file_search_queries": row.monthly_file_search_queries,
                    "storage_gb_limit": float(row.storage_gb_limit),
                    "monthly_price_usd": float(row.monthly_price_usd),
                    "features": row.features or {},
                    "is_active": row.is_active,
                }

                # Cache the result
                _tier_cache[tier_id] = (tier_data, time.time())
                logger.debug(f"Cached tier {tier_id}")

                return tier_data

        except Exception as e:
            logger.error(f"Failed to get tier {tier_id}: {e}")
            return None


# Module-level instance for convenience
_subscription_manager: Optional[SubscriptionManager] = None


def get_subscription_manager() -> SubscriptionManager:
    """Get or create SubscriptionManager instance."""
    global _subscription_manager
    if _subscription_manager is None:
        _subscription_manager = SubscriptionManager()
    return _subscription_manager
