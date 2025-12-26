"""
Short-term memory for conversation history within sessions.

Uses LangChain's InMemoryChatMessageHistory for storing messages
and provides session-based conversation management.
"""

import logging
from typing import Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """
    Manages short-term conversation memory within sessions.

    Provides:
    - Per-session message storage
    - Automatic message trimming when limits exceeded
    - LangChain-compatible chat history interface
    """

    def __init__(self, max_messages: int = 20, auto_summarize: bool = True):
        """
        Initialize short-term memory.

        Args:
            max_messages: Maximum messages to keep per session
            auto_summarize: Whether to auto-summarize on trim (future feature)
        """
        self.max_messages = max_messages
        self.auto_summarize = auto_summarize
        self._sessions: Dict[str, ChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create chat history for a session.

        This method is compatible with LangChain's RunnableWithMessageHistory.

        Args:
            session_id: Unique session identifier

        Returns:
            ChatMessageHistory for the session
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMessageHistory()
            logger.debug(f"Created new chat history for session: {session_id}")
        return self._sessions[session_id]

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """
        Get all messages for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of messages in the session
        """
        history = self.get_session_history(session_id)
        return history.messages

    def add_human_message(self, session_id: str, content: str) -> None:
        """
        Add a human message to the session.

        Args:
            session_id: Session identifier
            content: Message content
        """
        history = self.get_session_history(session_id)
        history.add_user_message(content)
        self._trim_if_needed(session_id)
        logger.debug(f"Added human message to session {session_id}")

    def add_ai_message(self, session_id: str, content: str) -> None:
        """
        Add an AI message to the session.

        Args:
            session_id: Session identifier
            content: Message content
        """
        history = self.get_session_history(session_id)
        history.add_ai_message(content)
        self._trim_if_needed(session_id)
        logger.debug(f"Added AI message to session {session_id}")

    def add_system_message(self, session_id: str, content: str) -> None:
        """
        Add a system message to the session.

        Args:
            session_id: Session identifier
            content: Message content
        """
        history = self.get_session_history(session_id)
        history.add_message(SystemMessage(content=content))
        logger.debug(f"Added system message to session {session_id}")

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """
        Add any message type to the session.

        Args:
            session_id: Session identifier
            message: LangChain message object
        """
        history = self.get_session_history(session_id)
        history.add_message(message)
        self._trim_if_needed(session_id)

    def _trim_if_needed(self, session_id: str) -> None:
        """
        Trim messages if session exceeds max limit.

        Preserves system message at start if present, keeps most recent messages.

        Args:
            session_id: Session identifier
        """
        history = self.get_session_history(session_id)
        messages = history.messages

        if len(messages) <= self.max_messages:
            return

        # Check if first message is system message
        has_system = messages and isinstance(messages[0], SystemMessage)

        if has_system:
            # Keep system message + most recent messages
            keep_count = self.max_messages - 1
            new_messages = [messages[0]] + messages[-keep_count:]
        else:
            # Keep most recent messages
            new_messages = messages[-self.max_messages:]

        # Replace history with trimmed messages
        history.clear()
        for msg in new_messages:
            history.add_message(msg)

        logger.debug(
            f"Trimmed session {session_id} from {len(messages)} to {len(new_messages)} messages"
        )

    def clear_session(self, session_id: str) -> None:
        """
        Clear all messages from a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()
            logger.debug(f"Cleared session: {session_id}")

    def delete_session(self, session_id: str) -> None:
        """
        Delete a session entirely.

        Args:
            session_id: Session identifier
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug(f"Deleted session: {session_id}")

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self._sessions)

    def get_message_count(self, session_id: str) -> int:
        """
        Get message count for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of messages in session
        """
        if session_id in self._sessions:
            return len(self._sessions[session_id].messages)
        return 0

    def get_conversation_summary(self, session_id: str) -> Optional[str]:
        """
        Get a simple summary of the conversation (last few exchanges).

        Args:
            session_id: Session identifier

        Returns:
            Brief summary string or None if no messages
        """
        messages = self.get_messages(session_id)
        if not messages:
            return None

        # Get last few human/AI exchanges
        summary_parts = []
        for msg in messages[-6:]:  # Last 3 exchanges
            if isinstance(msg, HumanMessage):
                summary_parts.append(f"User: {msg.content[:100]}...")
            elif isinstance(msg, AIMessage):
                summary_parts.append(f"AI: {msg.content[:100]}...")

        return "\n".join(summary_parts) if summary_parts else None
