"""Configuration for Sheets Agent.

Inherits shared settings from BaseAgentConfig and adds sheets-specific options.
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator

from src.agents.core.base_config import BaseAgentConfig
from src.constants import (
    DEFAULT_FILE_CACHE_SIZE,
    DEFAULT_PREVIEW_ROWS,
    DEFAULT_SAMPLE_ROWS,
    MAX_DISPLAY_ROWS,
    MIN_FILE_SIZE_MB,
    MAX_FILE_SIZE_MB,
)
from src.utils.env_utils import parse_bool_env, parse_int_env, parse_float_env


class SheetsAgentConfig(BaseAgentConfig):
    """Configuration settings for the Sheets Agent.

    Inherits from BaseAgentConfig:
    - temperature, timeout_seconds, max_retries
    - session_timeout_minutes
    - rate_limit_requests, rate_limit_window_seconds
    - allowed_file_base
    - enable_short_term_memory, enable_long_term_memory, short_term_max_messages
    - debug, log_level
    """

    # OpenAI LLM Configuration (sheets-specific)
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )

    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_SHEET_MODEL", "gpt-5.1-codex-mini"),
        description="OpenAI model to use for sheets agent"
    )

    # Override temperature with sheets-specific default
    temperature: float = Field(
        default_factory=lambda: parse_float_env("OPENAI_TEMPERATURE", 0.1),
        description="Temperature for LLM responses"
    )

    # File Configuration (sheets-specific)
    source_directory: str = Field(
        default_factory=lambda: os.getenv("SOURCE_DIRECTORY", "data/sheets"),
        description="Directory containing Excel/CSV files"
    )

    supported_extensions: List[str] = Field(
        default=[".xlsx", ".xls", ".csv", ".tsv"],
        description="Supported file extensions"
    )

    max_file_size_mb: int = Field(
        default_factory=lambda: parse_int_env("MAX_FILE_SIZE_MB", 100),
        description="Maximum file size in MB"
    )

    @field_validator('max_file_size_mb')
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if not MIN_FILE_SIZE_MB <= v <= MAX_FILE_SIZE_MB:
            raise ValueError(f"Max file size must be between {MIN_FILE_SIZE_MB} and {MAX_FILE_SIZE_MB} MB")
        return v

    # Agent Configuration (sheets-specific)
    max_tool_calls: int = Field(
        default_factory=lambda: parse_int_env("MAX_TOOL_CALLS", 5),
        description="Maximum number of tool calls per agent execution"
    )

    max_iterations: int = Field(
        default_factory=lambda: parse_int_env("MAX_ITERATIONS", 10),
        description="Maximum number of agent iterations"
    )

    # Performance Configuration (sheets-specific)
    max_result_rows: int = Field(
        default_factory=lambda: parse_int_env("MAX_RESULT_ROWS", 10000),
        description="Maximum rows returned from queries"
    )

    duckdb_pool_size: int = Field(
        default_factory=lambda: parse_int_env("DUCKDB_POOL_SIZE", 5),
        description="DuckDB connection pool size"
    )

    # Middleware Configuration (LangChain built-in middleware)
    enable_middleware: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_MIDDLEWARE", True),
        description="Enable LangChain middleware stack"
    )

    model_retry_max_attempts: int = Field(
        default_factory=lambda: parse_int_env("MODEL_RETRY_MAX_ATTEMPTS", 3),
        description="Maximum retry attempts for model calls"
    )

    tool_retry_max_attempts: int = Field(
        default_factory=lambda: parse_int_env("TOOL_RETRY_MAX_ATTEMPTS", 2),
        description="Maximum retry attempts for tool calls"
    )

    model_call_limit: int = Field(
        default_factory=lambda: parse_int_env("MODEL_CALL_LIMIT", 10),
        description="Maximum model calls per run"
    )

    tool_call_limit: int = Field(
        default_factory=lambda: parse_int_env("TOOL_CALL_LIMIT", 20),
        description="Maximum tool calls per run"
    )

    # Analysis Keywords (configurable to work with any dataset)
    financial_terms: List[str] = Field(
        default=['revenue', 'sales', 'income', 'profit', 'earnings', 'cost', 'expense', 'total', 'sum', 'amount'],
        description="Terms to identify financial/numeric columns"
    )

    quarterly_terms: List[str] = Field(
        default=['q1', 'q2', 'q3', 'q4', 'quarter'],
        description="Terms to identify quarterly columns"
    )

    temporal_terms: List[str] = Field(
        default=['trend', 'over time', 'monthly', 'yearly', 'growth', 'historical'],
        description="Terms indicating trend analysis"
    )

    display_keywords: List[str] = Field(
        default=['list', 'show', 'display', 'all items', 'all rows', 'view', 'get', 'line items'],
        description="Keywords for data display queries"
    )

    current_fiscal_year: str = Field(
        default_factory=lambda: os.getenv("CURRENT_FISCAL_YEAR", "25"),
        description="Current fiscal year (e.g., '25' for FY25)"
    )

    currency_symbol: str = Field(
        default_factory=lambda: os.getenv("CURRENCY_SYMBOL", "$"),
        description="Currency symbol for formatting"
    )

    # Preview/Sample sizes (centralized)
    preview_rows: int = Field(
        default=DEFAULT_PREVIEW_ROWS,
        description="Rows to show in file previews"
    )

    sample_rows: int = Field(
        default=DEFAULT_SAMPLE_ROWS,
        description="Rows to show in analysis samples"
    )

    max_display_rows: int = Field(
        default=MAX_DISPLAY_ROWS,
        description="Max rows for full data display"
    )

    file_cache_size: int = Field(
        default=DEFAULT_FILE_CACHE_SIZE,
        description="Number of files to cache in memory"
    )

    class Config:
        env_prefix = "SHEETS_AGENT_"
        case_sensitive = False
