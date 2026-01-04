"""
Memory configuration for Document and Sheets agents.

Provides environment-variable-controlled settings for:
- Short-term memory (conversation history within sessions)
- Long-term memory (persistent storage across sessions via PostgreSQL)
"""

from pydantic import BaseModel, Field

from src.utils.env_utils import parse_bool_env, parse_int_env


class MemoryConfig(BaseModel):
    """Configuration for agent memory systems."""

    # Short-term memory settings
    enable_short_term: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_SHORT_TERM_MEMORY", True),
        description="Enable short-term conversation memory"
    )
    max_messages: int = Field(
        default_factory=lambda: parse_int_env("SHORT_TERM_MEMORY_MAX_MESSAGES", 20),
        description="Maximum messages to keep in short-term memory"
    )
    auto_summarize: bool = Field(
        default_factory=lambda: parse_bool_env("SHORT_TERM_MEMORY_SUMMARIZE", True),
        description="Auto-summarize when message limit reached"
    )

    # Long-term memory settings (PostgreSQL)
    enable_long_term: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_LONG_TERM_MEMORY", True),
        description="Enable long-term persistent memory"
    )

    class Config:
        """Pydantic config."""
        validate_default = True
