"""
Long-term memory using PostgreSQL for persistent storage.

Provides:
- User preferences (persistent across sessions)
- Conversation summaries (historical context)
- Generic key-value storage with namespaces

Note: This module now uses PostgreSQL instead of Firestore.
The PostgresLongTermMemory class is imported from src.db.repositories.memory_repository.
"""

import logging
from typing import Optional

from .config import MemoryConfig
from .schemas import ConversationSummary, MemoryEntry, UserPreferences

logger = logging.getLogger(__name__)

# Import the PostgreSQL implementation
from src.db.repositories.memory_repository import PostgresLongTermMemory

__all__ = [
    "PostgresLongTermMemory",
]
