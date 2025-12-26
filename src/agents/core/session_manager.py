"""Thread-safe session manager for conversation continuity.

This module provides a shared SessionManager class that can be used by both
SheetsAgent and DocumentAgent to manage user sessions and response caching.
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from pydantic import BaseModel, Field

from ...constants import DEFAULT_SESSION_TIMEOUT_MINUTES, DEFAULT_RESPONSE_CACHE_SIZE

logger = logging.getLogger(__name__)


class SessionInfo(BaseModel):
    """Session information model.

    This model is shared between DocumentAgent and SheetsAgent.
    It includes all fields needed by both agents for compatibility.
    """

    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    expires_at: datetime = Field(..., description="Session expiration time")

    # Performance tracking
    query_count: int = Field(0, description="Number of queries in this session")
    total_tokens_used: int = Field(0, description="Total tokens used in session")
    total_processing_time_ms: float = Field(0, description="Total processing time for session")

    # Document agent context
    documents_processed: List[str] = Field(
        default_factory=list,
        description="Documents processed in this session"
    )

    # Sheets agent context (files_in_context is an alias for sheets compatibility)
    files_processed: List[str] = Field(
        default_factory=list,
        description="Files processed in this session"
    )
    files_in_context: List[str] = Field(
        default_factory=list,
        description="Files currently loaded in session (sheets agent)"
    )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class SessionManager:
    """
    Thread-safe session manager for conversation continuity.

    Manages user sessions with automatic expiration and response caching.
    Sessions are identified by unique session IDs and can store context
    about processed documents and cached responses.

    Attributes:
        timeout_minutes: Session timeout in minutes
        max_cache_size: Maximum cached responses per session

    Example:
        >>> manager = SessionManager(timeout_minutes=30)
        >>> session = manager.get_or_create_session()
        >>> print(session.session_id)
        '550e8400-e29b-41d4-a716-446655440000'
        >>> manager.update_session(session.session_id, query_count=1)
    """

    def __init__(
        self,
        timeout_minutes: int = DEFAULT_SESSION_TIMEOUT_MINUTES,
        max_cache_size: int = DEFAULT_RESPONSE_CACHE_SIZE,
    ):
        """
        Initialize the session manager.

        Args:
            timeout_minutes: Session timeout in minutes (default: 30)
            max_cache_size: Max cached responses per session (default: 10)
        """
        self.sessions: Dict[str, SessionInfo] = {}
        self.timeout_minutes = timeout_minutes
        self.max_cache_size = max_cache_size
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_or_create_session(
        self, session_id: Optional[str] = None
    ) -> SessionInfo:
        """
        Get existing session or create a new one.

        If an existing session is found and not expired, its activity
        timestamp is updated and timeout is extended. If expired or
        not found, a new session is created.

        Args:
            session_id: Optional existing session ID

        Returns:
            SessionInfo object for the session
        """
        with self._lock:
            if session_id and session_id in self.sessions:
                session = self.sessions[session_id]
                if datetime.now() < session.expires_at:
                    # Session still valid - update activity
                    session.last_activity = datetime.now()
                    session.expires_at = datetime.now() + timedelta(
                        minutes=self.timeout_minutes
                    )
                    return session
                else:
                    # Session expired - clean up
                    del self.sessions[session_id]
                    if session_id in self.response_cache:
                        del self.response_cache[session_id]

            # Create new session
            now = datetime.now()
            new_session = SessionInfo(
                session_id=session_id or str(uuid.uuid4()),
                created_at=now,
                last_activity=now,
                expires_at=now + timedelta(minutes=self.timeout_minutes),
            )
            self.sessions[new_session.session_id] = new_session
            return new_session

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session by ID without creating.

        Args:
            session_id: Session ID to look up

        Returns:
            SessionInfo if found and not expired, None otherwise
        """
        with self._lock:
            session = self.sessions.get(session_id)
            if session and datetime.now() < session.expires_at:
                return session
            return None

    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        Update session information.

        Args:
            session_id: Session ID to update
            **kwargs: Fields to update (e.g., query_count=5)

        Returns:
            True if session was updated, False if not found
        """
        with self._lock:
            if session_id not in self.sessions:
                return False

            session = self.sessions[session_id]
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.last_activity = datetime.now()
            return True

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            now = datetime.now()
            expired_sessions = [
                session_id
                for session_id, session in self.sessions.items()
                if now >= session.expires_at
            ]

            for session_id in expired_sessions:
                del self.sessions[session_id]
                if session_id in self.response_cache:
                    del self.response_cache[session_id]

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            return len(expired_sessions)

    def get_cached_response(
        self, session_id: str, query_hash: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a session and query.

        Args:
            session_id: Session ID
            query_hash: Hash of the query for cache lookup

        Returns:
            Cached response dict or None if not found
        """
        with self._lock:
            if session_id in self.response_cache:
                return self.response_cache[session_id].get(query_hash)
            return None

    def cache_response(
        self, session_id: str, query_hash: str, response: Dict[str, Any]
    ) -> None:
        """
        Cache a response for a session and query.

        Maintains cache size limit by removing oldest entries.

        Args:
            session_id: Session ID
            query_hash: Hash of the query for cache key
            response: Response data to cache
        """
        with self._lock:
            if session_id not in self.response_cache:
                self.response_cache[session_id] = {}

            # Evict oldest entry if at capacity
            if len(self.response_cache[session_id]) >= self.max_cache_size:
                oldest_key = next(iter(self.response_cache[session_id]))
                del self.response_cache[session_id][oldest_key]

            self.response_cache[session_id][query_hash] = response

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and its cache.

        Args:
            session_id: Session ID to delete

        Returns:
            True if session was deleted, False if not found
        """
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                if session_id in self.response_cache:
                    del self.response_cache[session_id]
                return True
            return False

    @property
    def active_count(self) -> int:
        """Get number of active (non-expired) sessions."""
        now = datetime.now()
        return sum(
            1 for s in self.sessions.values() if now < s.expires_at
        )

    def __repr__(self) -> str:
        return (
            f"SessionManager(timeout_minutes={self.timeout_minutes}, "
            f"active_sessions={self.active_count})"
        )
