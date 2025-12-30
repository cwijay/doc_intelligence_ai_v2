"""Configuration for Extractor Agent.

Inherits shared settings from BaseAgentConfig and adds extraction-specific options.
"""

import os
from typing import Optional
from pydantic import Field, field_validator

from src.agents.core.base_config import BaseAgentConfig


class ExtractorAgentConfig(BaseAgentConfig):
    """Configuration settings for the Extractor Agent.

    Inherits from BaseAgentConfig:
    - temperature, timeout_seconds, max_retries
    - session_timeout_minutes
    - rate_limit_requests, rate_limit_window_seconds
    - allowed_file_base
    - enable_short_term_memory, enable_long_term_memory, short_term_max_messages
    - debug, log_level
    """

    # Google API Key (for Gemini)
    google_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY"),
        description="Google API key for Gemini"
    )

    # Primary LLM: Gemini Pro
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("EXTRACTOR_AGENT_MODEL", "gemini-3-pro-preview"),
        description="Gemini model for extraction (primary)"
    )

    # Fallback LLM: Gemini Flash (faster, cheaper)
    gemini_fallback_model: str = Field(
        default_factory=lambda: os.getenv("EXTRACTOR_FALLBACK_MODEL", "gemini-3-flash-preview"),
        description="Gemini Flash model for fallback extraction"
    )

    # Override temperature with extraction-specific default (lower for structured output)
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("EXTRACTOR_AGENT_TEMPERATURE", "0.2")),
        description="Temperature for LLM responses (lower for more deterministic extraction)"
    )

    # Extraction Settings
    max_fields_to_analyze: int = Field(
        default_factory=lambda: int(os.getenv("EXTRACTOR_MAX_FIELDS", "50")),
        description="Maximum number of fields to analyze in a document"
    )

    @field_validator('max_fields_to_analyze')
    @classmethod
    def validate_max_fields(cls, v: int) -> int:
        if not 10 <= v <= 200:
            raise ValueError("max_fields_to_analyze must be between 10 and 200")
        return v

    max_schema_fields: int = Field(
        default_factory=lambda: int(os.getenv("EXTRACTOR_MAX_SCHEMA_FIELDS", "100")),
        description="Maximum fields allowed in a schema"
    )

    extraction_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("EXTRACTOR_TIMEOUT_SECONDS", "120")),
        description="Timeout for extraction operations"
    )

    @field_validator('extraction_timeout_seconds')
    @classmethod
    def validate_extraction_timeout(cls, v: int) -> int:
        if not 30 <= v <= 600:
            raise ValueError("extraction_timeout_seconds must be between 30 and 600")
        return v

    # Storage Settings
    schemas_directory: str = Field(
        default_factory=lambda: os.getenv("EXTRACTOR_SCHEMAS_DIR", "schemas"),
        description="GCS subfolder for extraction templates"
    )

    extracted_directory: str = Field(
        default_factory=lambda: os.getenv("EXTRACTOR_EXTRACTED_DIR", "extracted"),
        description="GCS subfolder for extracted data exports"
    )

    # Document Sources (reuse from environment)
    parsed_directory: str = Field(
        default_factory=lambda: os.getenv("PARSED_DIRECTORY", "parsed"),
        description="Directory containing pre-parsed .md files"
    )

    # Persistence
    persist_to_database: bool = Field(
        default_factory=lambda: os.getenv("EXTRACTOR_PERSIST", "true").lower() == "true",
        description="Whether to persist extracted data to database"
    )

    # Middleware Configuration (extraction-specific)
    enable_middleware: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_MIDDLEWARE", "true").lower() == "true",
        description="Enable LangChain middleware stack"
    )

    model_retry_max_attempts: int = Field(
        default_factory=lambda: int(os.getenv("MODEL_RETRY_MAX_ATTEMPTS", "3")),
        description="Maximum retry attempts for model calls"
    )

    tool_retry_max_attempts: int = Field(
        default_factory=lambda: int(os.getenv("TOOL_RETRY_MAX_ATTEMPTS", "2")),
        description="Maximum retry attempts for tool calls"
    )

    model_call_limit: int = Field(
        default_factory=lambda: int(os.getenv("MODEL_CALL_LIMIT", "15")),
        description="Maximum model calls per run"
    )

    tool_call_limit: int = Field(
        default_factory=lambda: int(os.getenv("TOOL_CALL_LIMIT", "10")),
        description="Maximum tool calls per run"
    )

    enable_pii_detection: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_PII_DETECTION", "true").lower() == "true",
        description="Enable PII detection middleware"
    )

    pii_strategy: str = Field(
        default_factory=lambda: os.getenv("PII_STRATEGY", "redact"),
        description="PII handling strategy: redact, mask, hash, or block"
    )

    @field_validator('pii_strategy')
    @classmethod
    def validate_pii_strategy(cls, v: str) -> str:
        valid = ['redact', 'mask', 'hash', 'block']
        if v.lower() not in valid:
            raise ValueError(f"pii_strategy must be one of: {valid}")
        return v.lower()

    class Config:
        env_prefix = "EXTRACTOR_AGENT_"
        case_sensitive = False
