"""
ResourceLogger - Handles non-token resource usage logging.

Single responsibility: Recording resource usage (pages, queries, storage).
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict

from sqlalchemy import text

logger = logging.getLogger(__name__)


# Mapping of resource types to database counter fields
RESOURCE_COUNTER_MAP = {
    "llamaparse_pages": "llamaparse_pages_used",
    "file_search_queries": "file_search_queries_used",
    "storage_bytes": "storage_used_bytes",
}


class ResourceLogger:
    """
    Logs non-token resource usage to database.

    Responsibilities:
    - Insert resource usage records
    - Update organization resource counters
    - Handle different resource types (pages, queries, storage)
    - Invalidate quota cache
    """

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
        """
        Log non-token resource usage.

        Args:
            org_id: Organization ID
            resource_type: 'llamaparse_pages', 'file_search_queries', 'storage_bytes'
            amount: Quantity used
            user_id: Optional user ID
            request_id: Optional request ID
            file_name: Optional file name
            file_path: Optional file path
            metadata: Optional metadata dict
        """
        try:
            from src.db.connection import db

            async with db.session() as session:
                if session is None:
                    logger.warning("Database disabled, skipping resource logging")
                    return

                # Insert resource record
                record_id = uuid.uuid4()
                await session.execute(
                    text("""
                        INSERT INTO resource_usage_records (
                            id, organization_id, user_id,
                            resource_type, amount,
                            request_id, file_name, file_path, metadata,
                            created_at
                        ) VALUES (
                            :id, :org_id, :user_id,
                            :resource_type, :amount,
                            :request_id, :file_name, :file_path, :metadata,
                            :created_at
                        )
                    """),
                    {
                        "id": record_id,
                        "org_id": org_id,
                        "user_id": user_id,
                        "resource_type": resource_type,
                        "amount": amount,
                        "request_id": request_id,
                        "file_name": file_name,
                        "file_path": file_path,
                        "metadata": json.dumps(metadata or {}),
                        "created_at": datetime.utcnow(),
                    }
                )

                # Update organization counter based on resource type
                if resource_type in RESOURCE_COUNTER_MAP:
                    counter_field = RESOURCE_COUNTER_MAP[resource_type]
                    await session.execute(
                        text(f"""
                            UPDATE organization_subscriptions
                            SET
                                {counter_field} = {counter_field} + :amount,
                                updated_at = :now
                            WHERE organization_id = :org_id
                        """),
                        {
                            "org_id": org_id,
                            "amount": amount,
                            "now": datetime.utcnow(),
                        }
                    )

                await session.commit()

                logger.info(
                    f"Logged resource usage: org={org_id}, "
                    f"type={resource_type}, amount={amount}"
                )

                # Invalidate quota cache
                self._invalidate_quota_cache(org_id)

        except Exception as e:
            logger.error(f"Failed to log resource usage for org {org_id}: {e}")

    def _invalidate_quota_cache(self, org_id: str) -> None:
        """Invalidate quota cache for organization."""
        try:
            from .quota_checker import get_quota_checker
            get_quota_checker().invalidate_cache(org_id)
        except Exception as e:
            logger.debug(f"Could not invalidate quota cache: {e}")


# Module-level instance for convenience
_resource_logger: Optional[ResourceLogger] = None


def get_resource_logger() -> ResourceLogger:
    """Get or create ResourceLogger instance."""
    global _resource_logger
    if _resource_logger is None:
        _resource_logger = ResourceLogger()
    return _resource_logger
