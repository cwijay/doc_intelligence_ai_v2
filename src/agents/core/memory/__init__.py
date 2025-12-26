"""
Memory module for Document and Sheets agents.

Provides short-term and long-term memory capabilities:

Short-term memory:
- Conversation history within sessions
- Automatic message trimming
- LangChain-compatible interface

Long-term memory:
- User preferences (persistent)
- Conversation summaries
- PostgreSQL-backed storage

Usage:
    from src.agents.core.memory import MemoryConfig, ShortTermMemory, PostgresLongTermMemory

    # Short-term memory
    short_term = ShortTermMemory(max_messages=20)
    short_term.add_human_message("session_123", "Hello!")
    history = short_term.get_messages("session_123")

    # Long-term memory
    config = MemoryConfig()
    long_term = PostgresLongTermMemory(config)
    prefs = long_term.get_or_create_preferences("user_456")
"""

from .config import MemoryConfig
from .long_term import PostgresLongTermMemory
from .schemas import (
    ConversationMessage,
    ConversationSummary,
    MemoryEntry,
    UserPreferences,
)
from .short_term import ShortTermMemory

__all__ = [
    # Config
    "MemoryConfig",
    # Short-term
    "ShortTermMemory",
    # Long-term
    "PostgresLongTermMemory",
    # Schemas
    "UserPreferences",
    "ConversationSummary",
    "MemoryEntry",
    "ConversationMessage",
]
