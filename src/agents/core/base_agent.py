"""Base agent class with shared functionality.

This module provides the BaseAgent abstract class that eliminates code
duplication between SheetsAgent and DocumentAgent.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .base_config import BaseAgentConfig
from .rate_limiter import RateLimiter
from .session_manager import SessionManager, SessionInfo

# Memory imports (optional - graceful fallback if not available)
try:
    from .memory import (
        MemoryConfig,
        ShortTermMemory,
        PostgresLongTermMemory,
        ConversationSummary
    )
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    ShortTermMemory = None
    PostgresLongTermMemory = None
    ConversationSummary = None
    MemoryConfig = None

# Token tracking imports (optional - graceful fallback if not available)
try:
    from src.core.usage import TokenTrackingCallbackHandler
    TOKEN_TRACKING_AVAILABLE = True
except ImportError:
    TOKEN_TRACKING_AVAILABLE = False
    TokenTrackingCallbackHandler = None

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Provides shared functionality for:
    - Session management
    - Rate limiting
    - Memory (short-term and long-term)
    - Audit logging
    - Token tracking callbacks
    - Health checks (base structure)
    - Session ending and cleanup

    Subclasses must implement:
    - _init_llm(): Initialize the language model
    - _init_tools(): Initialize agent tools
    - _create_agent(): Create the agent instance
    - _get_agent_type(): Return agent type string for audit/memory
    - _get_health_components(): Return agent-specific health components
    """

    def __init__(self, config: BaseAgentConfig):
        """Initialize base agent components.

        Args:
            config: Agent configuration (must inherit from BaseAgentConfig)
        """
        self.config = config

        # Initialize session manager and rate limiter
        self.session_manager = SessionManager(config.session_timeout_minutes)
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window_seconds
        )

        # Initialize memory systems
        self.short_term_memory: Optional[Any] = None
        self.long_term_memory: Optional[Any] = None
        self._init_memory()

        # Initialize audit logging
        self.audit_logger = self._init_audit_logging()

        # Token tracking callback (to be set up by subclass in _init_llm)
        self._token_callback = None

    def _init_memory(self) -> None:
        """Initialize short-term and long-term memory systems."""
        if not MEMORY_AVAILABLE:
            logger.warning("Memory module not available - memory features disabled")
            return

        # Initialize short-term memory
        if self.config.enable_short_term_memory:
            self.short_term_memory = ShortTermMemory(
                max_messages=self.config.short_term_max_messages
            )
            logger.info(
                f"Short-term memory enabled (max {self.config.short_term_max_messages} messages)"
            )

        # Initialize long-term memory (PostgreSQL-backed)
        if self.config.enable_long_term_memory:
            try:
                memory_config = MemoryConfig()
                self.long_term_memory = PostgresLongTermMemory(memory_config)
                logger.info("Long-term memory enabled (PostgreSQL)")
            except Exception as e:
                logger.warning(f"Failed to initialize long-term memory: {e}")

    def _init_audit_logging(self) -> Optional[callable]:
        """Initialize audit logging (PostgreSQL-backed).

        Returns:
            The log_event function if available, None otherwise.
        """
        try:
            from src.db.repositories.audit_repository import log_event
            logger.info("Initialized audit logging (PostgreSQL)")
            return log_event
        except ImportError as e:
            logger.warning(f"Audit module not available: {e}")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize audit logging: {e}")
            return None

    def _create_token_tracking_callback(self, feature: str) -> List:
        """Create token tracking callback for LLM.

        Args:
            feature: Feature name for tracking (e.g., 'sheets_agent', 'document_agent')

        Returns:
            List of callbacks to pass to LLM initialization
        """
        callbacks = []
        if TOKEN_TRACKING_AVAILABLE and TokenTrackingCallbackHandler:
            self._token_callback = TokenTrackingCallbackHandler(
                org_id="",  # Will use thread-local context
                feature=feature,
                use_context=True,  # Enable context-based org_id lookup
            )
            callbacks.append(self._token_callback)
            logger.debug("Token tracking callback initialized in context mode")
        return callbacks

    @abstractmethod
    def _init_llm(self) -> Any:
        """Initialize the language model.

        Must be implemented by subclasses to set up their specific LLM.

        Returns:
            The initialized LLM instance.
        """
        pass

    @abstractmethod
    def _init_tools(self) -> List:
        """Initialize agent tools.

        Must be implemented by subclasses to create their tool set.

        Returns:
            List of tools for the agent.
        """
        pass

    @abstractmethod
    def _create_agent(self) -> Any:
        """Create the agent instance.

        Must be implemented by subclasses to build their specific agent.

        Returns:
            The agent instance.
        """
        pass

    @abstractmethod
    def _get_agent_type(self) -> str:
        """Get the agent type identifier.

        Used for audit logging and memory summaries.

        Returns:
            Agent type string (e.g., 'sheets', 'document')
        """
        pass

    def end_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        save_summary: bool = True
    ) -> bool:
        """End a session and optionally save summary to long-term memory.

        Args:
            session_id: Session identifier
            user_id: User ID for long-term memory
            save_summary: Whether to save conversation summary

        Returns:
            True if session ended successfully
        """
        try:
            # Get session info
            session = self.session_manager.sessions.get(session_id)

            if not session:
                logger.warning(f"Session {session_id} not found")
                return False

            # Save conversation summary to long-term memory
            if save_summary and self.long_term_memory and user_id:
                self._save_conversation_summary(session_id, user_id, session)

            # Clear short-term memory for session
            if self.short_term_memory:
                self.short_term_memory.delete_session(session_id)

            # Remove from session manager
            if session_id in self.session_manager.sessions:
                del self.session_manager.sessions[session_id]

            logger.info(f"Session {session_id} ended successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to end session {session_id}: {e}")
            return False

    def _save_conversation_summary(
        self,
        session_id: str,
        user_id: str,
        session: SessionInfo
    ) -> None:
        """Save conversation summary to long-term memory.

        Args:
            session_id: Session identifier
            user_id: User ID for long-term memory
            session: Session info with query count and documents
        """
        if not self.long_term_memory or not MEMORY_AVAILABLE:
            return

        try:
            # Get conversation summary from short-term memory
            summary_text = ""
            if self.short_term_memory:
                summary_text = (
                    self.short_term_memory.get_conversation_summary(session_id)
                    or f"Session with {session.query_count} queries"
                )

            # Get documents discussed (handle both attribute names)
            documents = getattr(session, 'documents_processed', None) or \
                        getattr(session, 'files_in_context', [])

            summary = ConversationSummary(
                session_id=session_id,
                user_id=user_id,
                agent_type=self._get_agent_type(),
                summary=summary_text,
                key_topics=[],
                documents_discussed=documents,
                queries_count=session.query_count
            )

            self.long_term_memory.save_conversation_summary(summary)
            logger.info(f"Saved conversation summary for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to save conversation summary: {e}")

    def _get_base_health_status(self) -> Dict[str, Any]:
        """Get base health status components shared by all agents.

        Returns:
            Dict with common health status fields
        """
        # Cleanup expired sessions and rate limiter entries
        self.session_manager.cleanup_expired_sessions()
        self.rate_limiter.cleanup()

        return {
            "sessions": {
                "active_count": len(self.session_manager.sessions),
                "total_queries": sum(
                    s.query_count for s in self.session_manager.sessions.values()
                )
            },
            "rate_limiter": {
                "tracked_sessions": len(self.rate_limiter.requests)
            },
            "memory": {
                "short_term_sessions": (
                    self.short_term_memory.get_session_count()
                    if self.short_term_memory else 0
                ),
                "short_term_enabled": self.config.enable_short_term_memory,
                "long_term_enabled": self.config.enable_long_term_memory
            },
            "audit_logging": "healthy" if self.audit_logger else "disabled"
        }

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agent.

        Must be implemented by subclasses to add agent-specific components.
        Subclasses should call _get_base_health_status() and extend the result.

        Returns:
            Dict with health status information
        """
        pass

    def _cleanup_resources(self) -> None:
        """Cleanup shared resources during shutdown."""
        # Clean up expired sessions
        self.session_manager.cleanup_expired_sessions()

        # Clean up rate limiter
        self.rate_limiter.cleanup()

        logger.debug("Base agent resources cleaned up")

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown agent resources gracefully.

        Must be implemented by subclasses to handle agent-specific cleanup.
        Subclasses should call _cleanup_resources() as part of shutdown.

        Args:
            wait: If True, wait for pending tasks to complete.
        """
        pass
