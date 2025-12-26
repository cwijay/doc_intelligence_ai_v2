"""
Agents module.

Provides:
- DocumentAgent: Document analysis and content generation
- SheetsAgent: Spreadsheet data analysis
- core: Shared memory infrastructure

Note: Middleware functionality is now provided by LangChain 1.2.0 built-in middleware.
"""

from .document import DocumentAgent, DocumentAgentConfig
from .sheets import SheetsAgent, ChatRequest, ChatResponse, SheetsAgentConfig

# Re-export core components for convenience
from .core import (
    MemoryConfig,
    ShortTermMemory,
    PostgresLongTermMemory,
)

__all__ = [
    # Document Agent
    "DocumentAgent",
    "DocumentAgentConfig",
    # Sheets Agent
    "SheetsAgent",
    "SheetsAgentConfig",
    "ChatRequest",
    "ChatResponse",
    # Core (convenience exports)
    "MemoryConfig",
    "ShortTermMemory",
    "PostgresLongTermMemory",
]
