"""Base configuration for all agents.

Shared configuration fields are defined here to eliminate duplication
between SheetsAgentConfig and DocumentAgentConfig.
"""

import os
from pydantic import BaseModel, Field, field_validator

from src.constants import (
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    MIN_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    DEFAULT_AGENT_TIMEOUT_SECONDS,
    DEFAULT_SESSION_TIMEOUT_MINUTES,
    DEFAULT_RATE_LIMIT_REQUESTS,
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    PII_STRATEGIES,
    DEFAULT_PII_STRATEGY,
)
from src.utils.env_utils import parse_bool_env, parse_int_env


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
        if not MIN_TEMPERATURE <= v <= MAX_TEMPERATURE:
            raise ValueError(f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
        return v

    # Agent Timeout
    timeout_seconds: int = Field(
        default_factory=lambda: parse_int_env("AGENT_TIMEOUT", DEFAULT_AGENT_TIMEOUT_SECONDS),
        description="Timeout in seconds for agent operations"
    )

    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is within reasonable range."""
        if not MIN_TIMEOUT_SECONDS <= v <= MAX_TIMEOUT_SECONDS:
            raise ValueError(f"Timeout must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS} seconds")
        return v

    # Retry Configuration
    max_retries: int = Field(
        default_factory=lambda: parse_int_env("AGENT_MAX_RETRIES", 3),
        description="Maximum number of retries for agent operations"
    )

    # Session Configuration
    session_timeout_minutes: int = Field(
        default_factory=lambda: parse_int_env("SESSION_TIMEOUT_MINUTES", DEFAULT_SESSION_TIMEOUT_MINUTES),
        description="Session timeout in minutes"
    )

    # Rate Limiting Configuration
    rate_limit_requests: int = Field(
        default_factory=lambda: parse_int_env("RATE_LIMIT_REQUESTS", DEFAULT_RATE_LIMIT_REQUESTS),
        description="Maximum requests per rate limit window"
    )

    rate_limit_window_seconds: int = Field(
        default_factory=lambda: parse_int_env("RATE_LIMIT_WINDOW", DEFAULT_RATE_LIMIT_WINDOW_SECONDS),
        description="Rate limit window in seconds"
    )

    # Security Configuration
    allowed_file_base: str = Field(
        default_factory=lambda: os.getenv("ALLOWED_FILE_BASE", "/Users"),
        description="Base directory for allowed file access"
    )

    # Memory Configuration
    enable_short_term_memory: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_SHORT_TERM_MEMORY", True),
        description="Enable short-term conversation memory"
    )

    enable_long_term_memory: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_LONG_TERM_MEMORY", True),
        description="Enable long-term persistent memory"
    )

    short_term_max_messages: int = Field(
        default_factory=lambda: parse_int_env("SHORT_TERM_MEMORY_MAX_MESSAGES", 20),
        description="Maximum messages to keep in short-term memory"
    )

    # Development Configuration
    debug: bool = Field(
        default_factory=lambda: parse_bool_env("DEBUG", False),
        description="Enable debug mode"
    )

    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper(),
        description="Logging level"
    )

    # PII Detection Configuration (shared across agents)
    enable_pii_detection: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_PII_DETECTION", True),
        description="Enable PII detection middleware"
    )

    pii_strategy: str = Field(
        default_factory=lambda: os.getenv("PII_STRATEGY", DEFAULT_PII_STRATEGY),
        description="PII handling strategy: redact, mask, hash, or block"
    )

    @field_validator('pii_strategy')
    @classmethod
    def validate_pii_strategy(cls, v: str) -> str:
        """Validate PII strategy is one of the allowed values."""
        normalized = v.lower()
        if normalized not in PII_STRATEGIES:
            raise ValueError(f"pii_strategy must be one of: {list(PII_STRATEGIES)}")
        return normalized

    class Config:
        """Pydantic configuration."""
        case_sensitive = False
