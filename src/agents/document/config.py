"""Configuration for Document Agent.

Inherits shared settings from BaseAgentConfig and adds document-specific options.
"""

import os
from typing import Optional
from pydantic import Field, field_validator

from src.agents.core.base_config import BaseAgentConfig
from src.constants import (
    DEFAULT_NUM_FAQS,
    DEFAULT_NUM_QUESTIONS,
    DEFAULT_SUMMARY_MAX_WORDS,
    MIN_NUM_FAQS,
    MAX_NUM_FAQS,
    MIN_NUM_QUESTIONS,
    MAX_NUM_QUESTIONS,
    MIN_SUMMARY_WORDS,
    MAX_SUMMARY_WORDS,
    MIN_TOOL_SELECTOR_MAX_TOOLS,
    MAX_TOOL_SELECTOR_MAX_TOOLS,
)
from src.utils.env_utils import parse_bool_env, parse_int_env, parse_float_env


class DocumentAgentConfig(BaseAgentConfig):
    """Configuration settings for the Document Agent.

    Inherits from BaseAgentConfig:
    - temperature, timeout_seconds, max_retries
    - session_timeout_minutes
    - rate_limit_requests, rate_limit_window_seconds
    - allowed_file_base
    - enable_short_term_memory, enable_long_term_memory, short_term_max_messages
    - debug, log_level
    """

    # LLM Settings (OpenAI - document-specific)
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )

    openai_model: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_MODEL", "gpt-5-nano"),
        description="OpenAI model to use"
    )

    # Override temperature with document-specific default
    temperature: float = Field(
        default_factory=lambda: parse_float_env("DOCUMENT_AGENT_TEMPERATURE", 0.3),
        description="Temperature for LLM responses"
    )

    # Document Sources (document-specific)
    parsed_directory: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_PARSED_DIR", "parsed"),
        description="Directory containing pre-parsed .md files"
    )

    upload_directory: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_UPLOAD_DIR", "upload"),
        description="Directory containing raw text files (.txt, .md)"
    )

    # Generation Defaults (document-specific)
    default_num_faqs: int = Field(
        default_factory=lambda: parse_int_env("DOCUMENT_AGENT_NUM_FAQS", DEFAULT_NUM_FAQS),
        description="Default number of FAQs to generate"
    )

    default_num_questions: int = Field(
        default_factory=lambda: parse_int_env("DOCUMENT_AGENT_NUM_QUESTIONS", DEFAULT_NUM_QUESTIONS),
        description="Default number of questions to generate"
    )

    max_num_faqs: int = Field(
        default=MAX_NUM_FAQS,
        description="Maximum allowed FAQs"
    )

    max_num_questions: int = Field(
        default=MAX_NUM_QUESTIONS,
        description="Maximum allowed questions"
    )

    summary_max_words: int = Field(
        default_factory=lambda: parse_int_env("DOCUMENT_AGENT_SUMMARY_WORDS", DEFAULT_SUMMARY_MAX_WORDS),
        description="Maximum words for summary"
    )

    @field_validator('default_num_faqs', 'max_num_faqs')
    @classmethod
    def validate_faqs(cls, v: int) -> int:
        if not MIN_NUM_FAQS <= v <= MAX_NUM_FAQS:
            raise ValueError(f"FAQ count must be between {MIN_NUM_FAQS} and {MAX_NUM_FAQS}")
        return v

    @field_validator('default_num_questions', 'max_num_questions')
    @classmethod
    def validate_questions(cls, v: int) -> int:
        if not MIN_NUM_QUESTIONS <= v <= MAX_NUM_QUESTIONS:
            raise ValueError(f"Question count must be between {MIN_NUM_QUESTIONS} and {MAX_NUM_QUESTIONS}")
        return v

    @field_validator('summary_max_words')
    @classmethod
    def validate_summary_words(cls, v: int) -> int:
        if not MIN_SUMMARY_WORDS <= v <= MAX_SUMMARY_WORDS:
            raise ValueError(f"Summary max words must be between {MIN_SUMMARY_WORDS} and {MAX_SUMMARY_WORDS}")
        return v

    # Persistence (document-specific)
    persist_to_database: bool = Field(
        default_factory=lambda: parse_bool_env("DOCUMENT_AGENT_PERSIST", True),
        description="Whether to persist results to database"
    )

    output_directory: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_OUTPUT_DIR", "generated"),
        description="Directory for JSON output files"
    )

    # Middleware Configuration (document-specific)
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
        default_factory=lambda: parse_int_env("MODEL_CALL_LIMIT", 15),
        description="Maximum model calls per run"
    )

    tool_call_limit: int = Field(
        default_factory=lambda: parse_int_env("TOOL_CALL_LIMIT", 10),
        description="Maximum tool calls per run"
    )

    enable_pii_detection: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_PII_DETECTION", True),
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

    # Tool Selection Configuration (document-specific)
    enable_tool_selection: bool = Field(
        default_factory=lambda: parse_bool_env("ENABLE_TOOL_SELECTION", True),
        description="Enable intelligent tool pre-selection based on query intent"
    )

    tool_selector_model: str = Field(
        default_factory=lambda: os.getenv("TOOL_SELECTOR_MODEL", "gpt-5.2-2025-12-11"),
        description="Model for tool selection (gpt-5.2 for better accuracy)"
    )

    tool_selector_max_tools: int = Field(
        default_factory=lambda: parse_int_env("TOOL_SELECTOR_MAX_TOOLS", 3),
        description="Maximum tools to provide per query after filtering"
    )

    @field_validator('tool_selector_max_tools')
    @classmethod
    def validate_max_tools(cls, v: int) -> int:
        if not MIN_TOOL_SELECTOR_MAX_TOOLS <= v <= MAX_TOOL_SELECTOR_MAX_TOOLS:
            raise ValueError(f"tool_selector_max_tools must be between {MIN_TOOL_SELECTOR_MAX_TOOLS} and {MAX_TOOL_SELECTOR_MAX_TOOLS}")
        return v

    class Config:
        env_prefix = "DOCUMENT_AGENT_"
        case_sensitive = False
