"""Base configuration for all agents.

Shared configuration fields are defined here to eliminate duplication
between SheetsAgentConfig and DocumentAgentConfig.
"""

import os
from pydantic import BaseModel, Field, field_validator


class BaseAgentConfig(BaseModel):
    """Base configuration shared by all agents.

    Contains common settings for:
    - Temperature and timeout
    - Session management
    - Rate limiting
    - Memory (short-term and long-term)
    - Security and debugging
    """

    # LLM Temperature (shared validation logic)
    temperature: float = Field(
        default=0.1,
        description="Temperature for LLM responses (0.0-2.0)"
    )

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    # Agent Timeout
    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_TIMEOUT", "300")),
        description="Timeout in seconds for agent operations"
    )

    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is within reasonable range."""
        if not 10 <= v <= 600:
            raise ValueError("Timeout must be between 10 and 600 seconds")
        return v

    # Retry Configuration
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_RETRIES", "3")),
        description="Maximum number of retries for agent operations"
    )

    # Session Configuration
    session_timeout_minutes: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")),
        description="Session timeout in minutes"
    )

    # Rate Limiting Configuration
    rate_limit_requests: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "10")),
        description="Maximum requests per rate limit window"
    )

    rate_limit_window_seconds: int = Field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        description="Rate limit window in seconds"
    )

    # Security Configuration
    allowed_file_base: str = Field(
        default_factory=lambda: os.getenv("ALLOWED_FILE_BASE", "/Users"),
        description="Base directory for allowed file access"
    )

    # Memory Configuration
    enable_short_term_memory: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_SHORT_TERM_MEMORY", "true").lower() == "true",
        description="Enable short-term conversation memory"
    )

    enable_long_term_memory: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_LONG_TERM_MEMORY", "true").lower() == "true",
        description="Enable long-term persistent memory"
    )

    short_term_max_messages: int = Field(
        default_factory=lambda: int(os.getenv("SHORT_TERM_MEMORY_MAX_MESSAGES", "20")),
        description="Maximum messages to keep in short-term memory"
    )

    # Development Configuration
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true",
        description="Enable debug mode"
    )

    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper(),
        description="Logging level"
    )

    class Config:
        """Pydantic configuration."""
        case_sensitive = False
