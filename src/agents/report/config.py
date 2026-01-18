"""Configuration for Report Agent.

Inherits shared settings from BaseAgentConfig and adds report-specific options.
"""

import os
from typing import Optional
from pydantic import Field, field_validator

from src.agents.core.base_config import BaseAgentConfig
from src.utils.env_utils import parse_bool_env, parse_int_env, parse_float_env


class ReportAgentConfig(BaseAgentConfig):
    """Configuration settings for the Report Agent.

    Inherits from BaseAgentConfig:
    - temperature, timeout_seconds, max_retries
    - session_timeout_minutes
    - rate_limit_requests, rate_limit_window_seconds
    - allowed_file_base
    - enable_short_term_memory, enable_long_term_memory, short_term_max_messages
    - debug, log_level
    """

    # OpenAI API Key
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key for report generation"
    )

    # Primary LLM: GPT-5-mini (balanced cost and quality for insights)
    openai_model: str = Field(
        default_factory=lambda: os.getenv("REPORT_AGENT_MODEL", "gpt-5-mini"),
        description="OpenAI model for report insights generation"
    )

    # Fallback LLM for complex analysis
    openai_fallback_model: str = Field(
        default_factory=lambda: os.getenv("REPORT_FALLBACK_MODEL", "gpt-5.2-2025-12-11"),
        description="OpenAI model for fallback report generation"
    )

    # Override temperature with report-specific default
    temperature: float = Field(
        default_factory=lambda: parse_float_env("REPORT_AGENT_TEMPERATURE", 0.3),
        description="Temperature for LLM responses (balanced for insights)"
    )

    # Report Generation Settings
    max_documents_per_report: int = Field(
        default_factory=lambda: parse_int_env("REPORT_MAX_DOCUMENTS", 100),
        description="Maximum number of documents to include in a report"
    )

    @field_validator('max_documents_per_report')
    @classmethod
    def validate_max_documents(cls, v: int) -> int:
        if not 1 <= v <= 500:
            raise ValueError("max_documents_per_report must be between 1 and 500")
        return v

    max_records_per_report: int = Field(
        default_factory=lambda: parse_int_env("REPORT_MAX_RECORDS", 10000),
        description="Maximum extracted records to analyze in a report"
    )

    report_timeout_seconds: int = Field(
        default_factory=lambda: parse_int_env("REPORT_TIMEOUT_SECONDS", 300),
        description="Timeout for report generation operations"
    )

    @field_validator('report_timeout_seconds')
    @classmethod
    def validate_report_timeout(cls, v: int) -> int:
        if not 30 <= v <= 900:
            raise ValueError("report_timeout_seconds must be between 30 and 900")
        return v

    # Storage Settings
    reports_directory: str = Field(
        default_factory=lambda: os.getenv("REPORT_OUTPUT_DIR", "reports"),
        description="GCS subfolder for generated reports"
    )

    charts_directory: str = Field(
        default_factory=lambda: os.getenv("REPORT_CHARTS_DIR", "charts"),
        description="GCS subfolder for generated charts"
    )

    # Document Sources (reuse from environment)
    parsed_directory: str = Field(
        default_factory=lambda: os.getenv("PARSED_DIRECTORY", "parsed"),
        description="Directory containing pre-parsed .md files"
    )

    extracted_directory: str = Field(
        default_factory=lambda: os.getenv("EXTRACTOR_EXTRACTED_DIR", "extracted"),
        description="Directory containing extracted data"
    )

    # Persistence
    persist_to_database: bool = Field(
        default_factory=lambda: parse_bool_env("REPORT_PERSIST", True),
        description="Whether to persist reports to database"
    )

    persist_to_gcs: bool = Field(
        default_factory=lambda: parse_bool_env("REPORT_PERSIST_GCS", True),
        description="Whether to persist reports to GCS"
    )

    # Chart Generation
    enable_charts: bool = Field(
        default_factory=lambda: parse_bool_env("REPORT_ENABLE_CHARTS", True),
        description="Enable chart generation in reports"
    )

    chart_dpi: int = Field(
        default_factory=lambda: parse_int_env("REPORT_CHART_DPI", 150),
        description="DPI for generated chart images"
    )

    # Middleware Configuration
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
        default_factory=lambda: parse_int_env("MODEL_CALL_LIMIT", 20),
        description="Maximum model calls per report"
    )

    tool_call_limit: int = Field(
        default_factory=lambda: parse_int_env("TOOL_CALL_LIMIT", 15),
        description="Maximum tool calls per report"
    )

    class Config:
        env_prefix = "REPORT_AGENT_"
        case_sensitive = False
