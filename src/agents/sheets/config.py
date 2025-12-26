"""Configuration for Sheets Agent."""

import os
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SheetsAgentConfig(BaseModel):
    """Configuration settings for the Sheets Agent."""

    # OpenAI LLM Configuration
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )

    openai_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_SHEET_MODEL", "gpt-5.1-codex-mini"),
        description="OpenAI model to use for sheets agent"
    )

    temperature: float = Field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
        description="Temperature for LLM responses"
    )

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    # File Configuration
    source_directory: str = Field(
        default_factory=lambda: os.getenv("SOURCE_DIRECTORY", "data/sheets"),
        description="Directory containing Excel/CSV files"
    )

    supported_extensions: List[str] = Field(
        default=[".xlsx", ".xls", ".csv", ".tsv"],
        description="Supported file extensions"
    )

    max_file_size_mb: int = Field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "100")),
        description="Maximum file size in MB"
    )

    @field_validator('max_file_size_mb')
    @classmethod
    def validate_max_file_size(cls, v: int) -> int:
        if not 1 <= v <= 500:
            raise ValueError("Max file size must be between 1 and 500 MB")
        return v

    # Agent Configuration
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("SHEETS_AGENT_MAX_RETRIES", "3")),
        description="Maximum number of retries for agent operations"
    )

    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("SHEETS_AGENT_TIMEOUT", "300")),
        description="Timeout in seconds for agent operations"
    )

    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if not 10 <= v <= 600:
            raise ValueError("Timeout must be between 10 and 600 seconds")
        return v

    max_tool_calls: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS", "5")),
        description="Maximum number of tool calls per agent execution"
    )

    max_iterations: int = Field(
        default_factory=lambda: int(os.getenv("MAX_ITERATIONS", "10")),
        description="Maximum number of agent iterations"
    )

    # Session Configuration
    session_timeout_minutes: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_TIMEOUT_MINUTES", "30")),
        description="Session timeout in minutes"
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

    # Performance Configuration
    max_result_rows: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RESULT_ROWS", "10000")),
        description="Maximum rows returned from queries"
    )

    duckdb_pool_size: int = Field(
        default_factory=lambda: int(os.getenv("DUCKDB_POOL_SIZE", "5")),
        description="DuckDB connection pool size"
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

    class Config:
        env_prefix = "SHEETS_AGENT_"
        case_sensitive = False
