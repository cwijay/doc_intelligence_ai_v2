"""
TokenLogger - Handles token usage logging.

Single responsibility: Recording token usage and updating counters.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict

from sqlalchemy import text

from .schemas import TokenUsage, calculate_cost

logger = logging.getLogger(__name__)


class TokenLogger:
    """
    Logs token usage to database and updates organization counters.

    Responsibilities:
    - Insert token usage records
    - Update organization token counters
    - Calculate costs
    - Invalidate quota cache
    """

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
        """
        Log token usage and update organization counters.

        Args:
            org_id: Organization ID
            feature: Feature name (document_agent, sheets_agent, rag_search)
            usage: TokenUsage data
            user_id: Optional user ID
            request_id: Optional request ID for deduplication
            session_id: Optional session ID
            metadata: Optional metadata dict
            processing_time_ms: Optional processing time in ms
        """
        try:
            from src.db.connection import db

            # Calculate costs
            input_cost, output_cost, total_cost = calculate_cost(
                usage.model or "default",
                usage.input_tokens,
                usage.output_tokens,
            )

            async with db.session() as session:
                if session is None:
                    logger.warning("Database disabled, skipping usage logging")
                    return

                # Insert usage record
                record_id = uuid.uuid4()
                await session.execute(
                    text("""
                        INSERT INTO token_usage_records (
                            id, organization_id, user_id, request_id, session_id,
                            feature, provider, model,
                            input_tokens, output_tokens, total_tokens, cached_tokens,
                            input_cost_usd, output_cost_usd, total_cost_usd,
                            metadata, processing_time_ms, created_at
                        ) VALUES (
                            :id, :org_id, :user_id, :request_id, :session_id,
                            :feature, :provider, :model,
                            :input_tokens, :output_tokens, :total_tokens, :cached_tokens,
                            :input_cost, :output_cost, :total_cost,
                            :metadata, :processing_time_ms, :created_at
                        )
                        ON CONFLICT (request_id) DO NOTHING
                    """),
                    {
                        "id": record_id,
                        "org_id": org_id,
                        "user_id": user_id,
                        "request_id": request_id or str(uuid.uuid4()),
                        "session_id": session_id,
                        "feature": feature,
                        "provider": usage.provider,
                        "model": usage.model,
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total_tokens,
                        "cached_tokens": usage.cached_tokens,
                        "input_cost": input_cost,
                        "output_cost": output_cost,
                        "total_cost": total_cost,
                        "metadata": json.dumps(metadata or {}),
                        "processing_time_ms": processing_time_ms,
                        "created_at": datetime.utcnow(),
                    }
                )

                # Atomically increment organization counter
                await session.execute(
                    text("""
                        UPDATE organization_subscriptions
                        SET
                            tokens_used_this_period = tokens_used_this_period + :tokens,
                            updated_at = :now
                        WHERE organization_id = :org_id
                    """),
                    {
                        "org_id": org_id,
                        "tokens": usage.total_tokens,
                        "now": datetime.utcnow(),
                    }
                )

                await session.commit()

                logger.info(
                    f"Logged token usage: org={org_id}, feature={feature}, "
                    f"tokens={usage.total_tokens}, cost=${total_cost:.6f}"
                )

                # Invalidate quota cache
                self._invalidate_quota_cache(org_id)

        except Exception as e:
            logger.error(f"Failed to log token usage for org {org_id}: {e}")
            # Don't raise - usage logging should not break the main flow

    def _invalidate_quota_cache(self, org_id: str) -> None:
        """Invalidate quota cache for organization."""
        try:
            from .quota_checker import get_quota_checker
            get_quota_checker().invalidate_cache(org_id)
        except Exception as e:
            logger.debug(f"Could not invalidate quota cache: {e}")


# Module-level instance for convenience
_token_logger: Optional[TokenLogger] = None


def get_token_logger() -> TokenLogger:
    """Get or create TokenLogger instance."""
    global _token_logger
    if _token_logger is None:
        _token_logger = TokenLogger()
    return _token_logger
