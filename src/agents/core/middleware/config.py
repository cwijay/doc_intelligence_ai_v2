"""
Middleware configuration for DocumentAgent.

Provides environment-variable-controlled settings for:
- LLM Tool Selector
- Model/Tool Retry
- Model Fallback
- Call Limits
- PII Detection
"""

import os
from pydantic import BaseModel, Field

from src.utils.env_utils import parse_bool_env, parse_int_env, parse_float_env


class MiddlewareConfig(BaseModel):
    """Configuration for all middleware components."""

    # Master switch
    enabled: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_MIDDLEWARE", True),
        description="Master switch to enable/disable all middleware"
    )

    # Tool Selector
    tool_selector_enabled: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_TOOL_SELECTOR", True),
        description="Enable LLM-based tool selection"
    )
    tool_selector_model: str = Field(
        default_factory=lambda: os.getenv("TOOL_SELECTOR_MODEL", "gpt-5-nano"),
        description="Lightweight model for tool selection"
    )
    tool_selector_max_tools: int = Field(
        default_factory=lambda: parse_int_env("TOOL_SELECTOR_MAX_TOOLS", 3),
        description="Maximum tools to select per query"
    )

    # Model Retry
    model_retry_enabled: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_MODEL_RETRY", True),
        description="Enable model retry with exponential backoff"
    )
    model_retry_max_attempts: int = Field(
        default_factory=lambda: parse_int_env("MODEL_RETRY_MAX_ATTEMPTS", 3),
        description="Maximum retry attempts for model calls"
    )
    model_retry_initial_delay: float = Field(
        default_factory=lambda: parse_float_env("MODEL_RETRY_INITIAL_DELAY", 1.0),
        description="Initial delay in seconds between retries"
    )
    model_retry_max_delay: float = Field(
        default_factory=lambda: parse_float_env("MODEL_RETRY_MAX_DELAY", 10.0),
        description="Maximum delay in seconds between retries"
    )

    # Tool Retry
    tool_retry_enabled: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_TOOL_RETRY", True),
        description="Enable tool retry for failed executions"
    )
    tool_retry_max_attempts: int = Field(
        default_factory=lambda: parse_int_env("TOOL_RETRY_MAX_ATTEMPTS", 2),
        description="Maximum retry attempts for tool calls"
    )
    tool_retry_delay: float = Field(
        default_factory=lambda: parse_float_env("TOOL_RETRY_DELAY", 1.0),
        description="Delay in seconds between tool retries"
    )

    # Model Fallback
    fallback_enabled: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_MODEL_FALLBACK", True),
        description="Enable fallback to alternative model on failure"
    )
    fallback_model: str = Field(
        default_factory=lambda: os.getenv("FALLBACK_MODEL", "gpt-5.2-2025-12-11"),
        description="Fallback model to use when primary fails"
    )
    fallback_provider: str = Field(
        default_factory=lambda: os.getenv("FALLBACK_PROVIDER", "openai"),
        description="Provider for fallback model"
    )

    # Call Limits
    model_call_limit: int = Field(
        default_factory=lambda: parse_int_env("MODEL_CALL_LIMIT_PER_RUN", 15),
        description="Maximum model calls per run"
    )
    tool_call_limit: int = Field(
        default_factory=lambda: parse_int_env("TOOL_CALL_LIMIT_PER_RUN", 10),
        description="Maximum tool calls per run"
    )

    # PII Detection
    pii_enabled: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_PII_DETECTION", True),
        description="Enable PII detection and handling"
    )
    pii_strategy: str = Field(
        default_factory=lambda: os.getenv("PII_STRATEGY", "redact"),
        description="PII handling strategy: redact, mask, hash, or block"
    )

    class Config:
        """Pydantic config."""
        validate_default = True
