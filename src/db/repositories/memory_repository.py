"""
Long-term memory repository - PostgreSQL implementation.

Multi-tenancy: All operations are scoped by organization_id for tenant isolation.

Provides PostgresLongTermMemory class as a drop-in replacement for
FirestoreLongTermMemory from src/agents/core/memory/long_term.py.

Stores:
- User preferences
- Conversation summaries
- Generic key-value memory entries
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select, delete, and_, desc
from sqlalchemy.dialects.postgresql import insert

from ..models import (
    UserPreference,
    ConversationSummary as ConversationSummaryModel,
    MemoryEntry as MemoryEntryModel,
)
from ..connection import db
from ..utils import with_db_retry

logger = logging.getLogger(__name__)


class PostgresLongTermMemory:
    """
    PostgreSQL-backed long-term memory store.

    Multi-tenancy: Supports organization_id for tenant isolation.

    API-compatible replacement for FirestoreLongTermMemory.
    Stores:
    - User preferences: user_preferences table
    - Conversation summaries: conversation_summaries table
    - Generic entries: memory_entries table
    """

    def __init__(
        self,
        config: Optional["MemoryConfig"] = None,
        organization_id: Optional[str] = None,
    ):
        """
        Initialize PostgreSQL long-term memory.

        Args:
            config: Memory configuration (uses defaults if not provided)
            organization_id: Organization ID for tenant isolation
        """
        # Import here to avoid circular imports
        from src.agents.core.memory.config import MemoryConfig

        self.config = config or MemoryConfig()
        self.organization_id = organization_id
        logger.info(f"Initialized PostgreSQL long-term memory org={organization_id}")

    # === User Preferences ===

    @with_db_retry
    async def get_user_preferences(self, user_id: str) -> Optional["UserPreferences"]:
        """
        Retrieve user preferences from PostgreSQL.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            user_id: User identifier

        Returns:
            UserPreferences if found, None otherwise
        """
        from src.agents.core.memory.schemas import UserPreferences

        async with db.session() as session:
            where_clauses = [UserPreference.user_id == user_id]
            if self.organization_id:
                where_clauses.append(UserPreference.organization_id == self.organization_id)

            stmt = select(UserPreference).where(and_(*where_clauses))
            result = await session.execute(stmt)
            prefs = result.scalar_one_or_none()

            if prefs:
                return UserPreferences(
                    user_id=prefs.user_id,
                    organization_id=prefs.organization_id,
                    preferred_language=prefs.preferred_language,
                    preferred_summary_length=prefs.preferred_summary_length,
                    preferred_faq_count=prefs.preferred_faq_count,
                    preferred_question_count=prefs.preferred_question_count,
                    custom_settings=prefs.custom_settings,
                    created_at=prefs.created_at,
                    updated_at=prefs.updated_at,
                )
            return None

    @with_db_retry
    async def save_user_preferences(self, preferences: "UserPreferences") -> None:
        """
        Save user preferences to PostgreSQL using upsert.

        Multi-tenancy: Associates preferences with organization_id set on instance.

        Args:
            preferences: UserPreferences object to save
        """
        async with db.session() as session:
            stmt = (
                insert(UserPreference)
                .values(
                    user_id=preferences.user_id,
                    organization_id=self.organization_id,
                    preferred_language=preferences.preferred_language or "en",
                    preferred_summary_length=preferences.preferred_summary_length or 500,
                    preferred_faq_count=preferences.preferred_faq_count or 5,
                    preferred_question_count=preferences.preferred_question_count or 10,
                    custom_settings=preferences.custom_settings or {},
                    created_at=preferences.created_at or datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["user_id"],
                    set_={
                        "organization_id": self.organization_id,
                        "preferred_language": preferences.preferred_language or "en",
                        "preferred_summary_length": preferences.preferred_summary_length or 500,
                        "preferred_faq_count": preferences.preferred_faq_count or 5,
                        "preferred_question_count": preferences.preferred_question_count or 10,
                        "custom_settings": preferences.custom_settings or {},
                        "updated_at": datetime.utcnow(),
                    },
                )
            )
            await session.execute(stmt)
            logger.info(f"Saved preferences for user: {preferences.user_id} org={self.organization_id}")

    @with_db_retry
    async def update_user_preference(
        self, user_id: str, key: str, value: Any
    ) -> None:
        """
        Update a single user preference.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            user_id: User identifier
            key: Preference key to update
            value: New value
        """
        async with db.session() as session:
            where_clauses = [UserPreference.user_id == user_id]
            if self.organization_id:
                where_clauses.append(UserPreference.organization_id == self.organization_id)

            stmt = select(UserPreference).where(and_(*where_clauses))
            result = await session.execute(stmt)
            prefs = result.scalar_one_or_none()

            if prefs:
                setattr(prefs, key, value)
                prefs.updated_at = datetime.utcnow()
                logger.debug(f"Updated preference {key} for user {user_id} org={self.organization_id}")

    async def get_or_create_preferences(self, user_id: str) -> "UserPreferences":
        """
        Get existing preferences or create defaults.

        Args:
            user_id: User identifier

        Returns:
            UserPreferences (existing or new defaults)
        """
        from src.agents.core.memory.schemas import UserPreferences

        prefs = await self.get_user_preferences(user_id)
        if prefs is None:
            prefs = UserPreferences(user_id=user_id)
            await self.save_user_preferences(prefs)
            logger.info(f"Created default preferences for user: {user_id}")
            prefs = await self.get_user_preferences(user_id)
        return prefs

    # === Conversation Summaries ===

    @with_db_retry
    async def save_conversation_summary(
        self, summary: "ConversationSummary"
    ) -> None:
        """
        Save a conversation summary using upsert.

        Multi-tenancy: Associates summary with organization_id set on instance.

        Args:
            summary: ConversationSummary to save
        """
        async with db.session() as session:
            stmt = (
                insert(ConversationSummaryModel)
                .values(
                    session_id=summary.session_id,
                    organization_id=self.organization_id,
                    user_id=summary.user_id or "anonymous",
                    agent_type=summary.agent_type,
                    summary=summary.summary,
                    key_topics=summary.key_topics or [],
                    documents_discussed=summary.documents_discussed or [],
                    queries_count=summary.queries_count,
                    created_at=summary.created_at or datetime.utcnow(),
                )
                .on_conflict_do_update(
                    index_elements=["session_id"],
                    set_={
                        "organization_id": self.organization_id,
                        "summary": summary.summary,
                        "key_topics": summary.key_topics or [],
                        "documents_discussed": summary.documents_discussed or [],
                        "queries_count": summary.queries_count,
                    },
                )
            )
            await session.execute(stmt)
            logger.info(f"Saved conversation summary for session: {summary.session_id} org={self.organization_id}")

    @with_db_retry
    async def get_conversation_summary(
        self, session_id: str
    ) -> Optional["ConversationSummary"]:
        """
        Get a specific conversation summary.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            session_id: Session identifier

        Returns:
            ConversationSummary if found, None otherwise
        """
        from src.agents.core.memory.schemas import ConversationSummary

        async with db.session() as session:
            where_clauses = [ConversationSummaryModel.session_id == session_id]
            if self.organization_id:
                where_clauses.append(ConversationSummaryModel.organization_id == self.organization_id)

            stmt = select(ConversationSummaryModel).where(and_(*where_clauses))
            result = await session.execute(stmt)
            summary_model = result.scalar_one_or_none()

            if summary_model:
                return ConversationSummary(
                    session_id=summary_model.session_id,
                    organization_id=summary_model.organization_id,
                    user_id=summary_model.user_id,
                    agent_type=summary_model.agent_type,
                    summary=summary_model.summary,
                    key_topics=list(summary_model.key_topics),
                    documents_discussed=list(summary_model.documents_discussed),
                    queries_count=summary_model.queries_count,
                    created_at=summary_model.created_at,
                )
            return None

    @with_db_retry
    async def get_user_summaries(
        self, user_id: str, limit: int = 10, agent_type: Optional[str] = None
    ) -> List["ConversationSummary"]:
        """
        Get recent conversation summaries for a user.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            user_id: User identifier
            limit: Maximum number of summaries to return
            agent_type: Filter by agent type ('document' or 'sheets')

        Returns:
            List of ConversationSummary objects (newest first)
        """
        from src.agents.core.memory.schemas import ConversationSummary

        async with db.session() as session:
            conditions = [ConversationSummaryModel.user_id == user_id]
            if agent_type:
                conditions.append(ConversationSummaryModel.agent_type == agent_type)
            if self.organization_id:
                conditions.append(ConversationSummaryModel.organization_id == self.organization_id)

            stmt = (
                select(ConversationSummaryModel)
                .where(and_(*conditions))
                .order_by(desc(ConversationSummaryModel.created_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            summaries = result.scalars().all()

            return [
                ConversationSummary(
                    session_id=s.session_id,
                    organization_id=s.organization_id,
                    user_id=s.user_id,
                    agent_type=s.agent_type,
                    summary=s.summary,
                    key_topics=list(s.key_topics),
                    documents_discussed=list(s.documents_discussed),
                    queries_count=s.queries_count,
                    created_at=s.created_at,
                )
                for s in summaries
            ]

    async def get_relevant_context(
        self, user_id: str, query: str, limit: int = 3
    ) -> str:
        """
        Get relevant context from past conversations for a query.

        Args:
            user_id: User identifier
            query: Current query for context matching
            limit: Maximum number of summaries to include

        Returns:
            Formatted context string or empty string if no history
        """
        summaries = await self.get_user_summaries(user_id, limit=limit)

        if not summaries:
            return ""

        context_parts = ["Relevant context from previous conversations:"]
        for s in summaries:
            topics = ", ".join(s.key_topics[:3]) if s.key_topics else "general"
            context_parts.append(f"- [{topics}] {s.summary[:200]}")

        return "\n".join(context_parts)

    # === Generic Memory Operations ===

    @with_db_retry
    async def put(self, namespace: str, key: str, data: Dict[str, Any]) -> None:
        """
        Store a generic memory entry using upsert.

        Multi-tenancy: Associates entry with organization_id set on instance.

        Args:
            namespace: Memory namespace
            key: Entry key
            data: Data to store
        """
        async with db.session() as session:
            stmt = (
                insert(MemoryEntryModel)
                .values(
                    organization_id=self.organization_id,
                    namespace=namespace,
                    key=key,
                    data=data,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
                .on_conflict_do_update(
                    constraint="uq_memory_org_namespace_key",
                    set_={
                        "data": data,
                        "updated_at": datetime.utcnow(),
                    },
                )
            )
            await session.execute(stmt)
            logger.debug(f"Stored memory entry: {namespace}/{key} org={self.organization_id}")

    @with_db_retry
    async def get(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a generic memory entry.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            namespace: Memory namespace
            key: Entry key

        Returns:
            Data dict if found, None otherwise
        """
        async with db.session() as session:
            where_clauses = [
                MemoryEntryModel.namespace == namespace,
                MemoryEntryModel.key == key,
            ]
            if self.organization_id:
                where_clauses.append(MemoryEntryModel.organization_id == self.organization_id)

            stmt = select(MemoryEntryModel).where(and_(*where_clauses))
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()

            if entry:
                return entry.data
            return None

    @with_db_retry
    async def delete(self, namespace: str, key: str) -> bool:
        """
        Delete a memory entry.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            namespace: Memory namespace
            key: Entry key

        Returns:
            True if deleted, False if not found
        """
        async with db.session() as session:
            where_clauses = [
                MemoryEntryModel.namespace == namespace,
                MemoryEntryModel.key == key,
            ]
            if self.organization_id:
                where_clauses.append(MemoryEntryModel.organization_id == self.organization_id)

            stmt = delete(MemoryEntryModel).where(and_(*where_clauses))
            result = await session.execute(stmt)

            if result.rowcount > 0:
                logger.debug(f"Deleted memory entry: {namespace}/{key} org={self.organization_id}")
                return True
            return False

    @with_db_retry
    async def search(
        self, namespace: str, filter_dict: Optional[Dict] = None, limit: int = 100
    ) -> List[Dict]:
        """
        Search memory entries in a namespace.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            namespace: Namespace to search
            filter_dict: Optional filters to apply on data fields
            limit: Maximum results

        Returns:
            List of matching entries
        """
        async with db.session() as session:
            where_clauses = [MemoryEntryModel.namespace == namespace]
            if self.organization_id:
                where_clauses.append(MemoryEntryModel.organization_id == self.organization_id)

            stmt = (
                select(MemoryEntryModel)
                .where(and_(*where_clauses))
                .limit(limit)
            )
            result = await session.execute(stmt)
            entries = result.scalars().all()

            results = []
            for entry in entries:
                entry_dict = {
                    "id": str(entry.id) if entry.id else None,
                    "organization_id": entry.organization_id,
                    "namespace": entry.namespace,
                    "key": entry.key,
                    "data": entry.data,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                }

                if filter_dict:
                    # Check if all filter conditions match
                    if all(entry.data.get(k) == v for k, v in filter_dict.items()):
                        results.append(entry_dict)
                else:
                    results.append(entry_dict)

            return results

    @with_db_retry
    async def clear_user_data(self, user_id: str) -> int:
        """
        Clear all memory entries for a user.

        Multi-tenancy: Filtered by organization_id set on instance.

        Args:
            user_id: User identifier

        Returns:
            Number of entries deleted
        """
        deleted = 0

        async with db.session() as session:
            # Delete preferences
            where_clauses = [UserPreference.user_id == user_id]
            if self.organization_id:
                where_clauses.append(UserPreference.organization_id == self.organization_id)
            stmt = delete(UserPreference).where(and_(*where_clauses))
            result = await session.execute(stmt)
            deleted += result.rowcount

            # Delete summaries
            where_clauses = [ConversationSummaryModel.user_id == user_id]
            if self.organization_id:
                where_clauses.append(ConversationSummaryModel.organization_id == self.organization_id)
            stmt = delete(ConversationSummaryModel).where(and_(*where_clauses))
            result = await session.execute(stmt)
            deleted += result.rowcount

        logger.info(f"Cleared {deleted} memory entries for user: {user_id} org={self.organization_id}")
        return deleted
