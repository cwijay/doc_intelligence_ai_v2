"""Configuration for Document Agent."""

import os
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DocumentAgentConfig(BaseModel):
    """Configuration settings for the Document Agent."""

    # LLM Settings (Gemini-only for cost efficiency)
    google_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY"),
        description="Google API key for Gemini"
    )

    gemini_model: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_MODEL", "gemini-3-flash-preview"),
        description="Gemini model to use"
    )

    temperature: float = Field(
        default_factory=lambda: float(os.getenv("DOCUMENT_AGENT_TEMPERATURE", "0.3")),
        description="Temperature for LLM responses"
    )

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    # Document Sources
    parsed_directory: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_PARSED_DIR", "parsed"),
        description="Directory containing pre-parsed .md files"
    )

    upload_directory: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_UPLOAD_DIR", "upload"),
        description="Directory containing raw text files (.txt, .md)"
    )

    # Generation Defaults (all configurable via env vars)
    default_num_faqs: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_NUM_FAQS", "10")),
        description="Default number of FAQs to generate"
    )

    default_num_questions: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_NUM_QUESTIONS", "10")),
        description="Default number of questions to generate"
    )

    max_num_faqs: int = Field(
        default=20,
        description="Maximum allowed FAQs"
    )

    max_num_questions: int = Field(
        default=50,
        description="Maximum allowed questions"
    )

    summary_max_words: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_SUMMARY_WORDS", "500")),
        description="Maximum words for summary"
    )

    @field_validator('default_num_faqs', 'max_num_faqs')
    @classmethod
    def validate_faqs(cls, v: int) -> int:
        if not 1 <= v <= 50:
            raise ValueError("FAQ count must be between 1 and 50")
        return v

    @field_validator('default_num_questions', 'max_num_questions')
    @classmethod
    def validate_questions(cls, v: int) -> int:
        if not 1 <= v <= 100:
            raise ValueError("Question count must be between 1 and 100")
        return v

    @field_validator('summary_max_words')
    @classmethod
    def validate_summary_words(cls, v: int) -> int:
        if not 50 <= v <= 2000:
            raise ValueError("Summary max words must be between 50 and 2000")
        return v

    # Agent Settings
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_MAX_RETRIES", "3")),
        description="Maximum number of retries for agent operations"
    )

    timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_TIMEOUT", "300")),
        description="Timeout in seconds for agent operations"
    )

    @field_validator('timeout_seconds')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if not 10 <= v <= 600:
            raise ValueError("Timeout must be between 10 and 600 seconds")
        return v

    session_timeout_minutes: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_SESSION_TIMEOUT", "30")),
        description="Session timeout in minutes"
    )

    # Persistence
    persist_to_database: bool = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_PERSIST", "true").lower() == "true",
        description="Whether to persist results to database"
    )

    output_directory: str = Field(
        default_factory=lambda: os.getenv("DOCUMENT_AGENT_OUTPUT_DIR", "generated"),
        description="Directory for JSON output files"
    )

    # Rate Limiting Configuration
    rate_limit_requests: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_RATE_LIMIT", "10")),
        description="Maximum requests per rate limit window"
    )

    rate_limit_window_seconds: int = Field(
        default_factory=lambda: int(os.getenv("DOCUMENT_AGENT_RATE_WINDOW", "60")),
        description="Rate limit window in seconds"
    )

    # Security Configuration
    allowed_file_base: str = Field(
        default_factory=lambda: os.getenv("ALLOWED_FILE_BASE", "/Users"),
        description="Base directory for allowed file access"
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

    # Middleware Configuration (LangChain 1.2.0 built-in)
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

    # Tool Selection Configuration
    enable_tool_selection: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_TOOL_SELECTION", "true").lower() == "true",
        description="Enable intelligent tool pre-selection based on query intent"
    )

    tool_selector_model: str = Field(
        default_factory=lambda: os.getenv("TOOL_SELECTOR_MODEL", "gemini-3-flash-preview"),
        description="Model for tool selection (should be fast/cheap)"
    )

    tool_selector_max_tools: int = Field(
        default_factory=lambda: int(os.getenv("TOOL_SELECTOR_MAX_TOOLS", "3")),
        description="Maximum tools to provide per query after filtering"
    )

    @field_validator('tool_selector_max_tools')
    @classmethod
    def validate_max_tools(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError("tool_selector_max_tools must be between 1 and 10")
        return v

    class Config:
        env_prefix = "DOCUMENT_AGENT_"
        case_sensitive = False
