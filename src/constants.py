"""Application-wide constants and configuration defaults.

This module centralizes magic values, default configurations, and constants
that are used across the codebase to improve maintainability.
"""

import os

# =============================================================================
# GCS URI Parsing Constants
# =============================================================================
GCS_URI_PREFIX = "gs://"
GCS_URI_PREFIX_LEN = len(GCS_URI_PREFIX)  # 5

# =============================================================================
# Content Preview Settings
# =============================================================================
DEFAULT_PREVIEW_LENGTH = 500

# =============================================================================
# Cache Configuration
# =============================================================================
DEFAULT_FILE_CACHE_SIZE = 50
DEFAULT_RESPONSE_CACHE_SIZE = 10

# =============================================================================
# Chunking Configuration (Gemini File Store)
# =============================================================================
DEFAULT_CHUNK_MAX_TOKENS = int(os.getenv("MAX_TOKENS_PER_CHUNK", "512"))
DEFAULT_CHUNK_OVERLAP_TOKENS = int(os.getenv("MAX_OVERLAP_TOKENS", "100"))

# =============================================================================
# Session & Rate Limiting
# =============================================================================
DEFAULT_SESSION_TIMEOUT_MINUTES = 30
DEFAULT_RATE_LIMIT_REQUESTS = 10
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60

# =============================================================================
# Agent Timeouts
# =============================================================================
DEFAULT_AGENT_TIMEOUT_SECONDS = 300
DEFAULT_CLEANUP_INTERVAL_SECONDS = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))

# =============================================================================
# Database Pool Configuration
# =============================================================================
DEFAULT_DB_POOL_SIZE = 5
DEFAULT_DB_MAX_OVERFLOW = 10
DEFAULT_DB_POOL_TIMEOUT = 30
DEFAULT_DB_POOL_RECYCLE = 1800  # 30 minutes in seconds

# =============================================================================
# Upload Configuration
# =============================================================================
DEFAULT_MAX_UPLOAD_SIZE_MB = 50

# =============================================================================
# Content Generation Defaults
# =============================================================================
DEFAULT_NUM_FAQS = 10
DEFAULT_NUM_QUESTIONS = 10
DEFAULT_SUMMARY_MAX_WORDS = 500

# =============================================================================
# Validation Bounds
# =============================================================================
MIN_NUM_FAQS = 1
MAX_NUM_FAQS = 50
MIN_NUM_QUESTIONS = 1
MAX_NUM_QUESTIONS = 100
MIN_SUMMARY_WORDS = 50
MAX_SUMMARY_WORDS = 2000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TIMEOUT_SECONDS = 10
MAX_TIMEOUT_SECONDS = 600
MIN_FILE_SIZE_MB = 1
MAX_FILE_SIZE_MB = 500
MIN_TOOL_SELECTOR_MAX_TOOLS = 1
MAX_TOOL_SELECTOR_MAX_TOOLS = 10

# =============================================================================
# Preview/Display Settings (Sheets Agent)
# =============================================================================
DEFAULT_PREVIEW_ROWS = 5
DEFAULT_SAMPLE_ROWS = 3
MAX_DISPLAY_ROWS = 100

# =============================================================================
# Cache TTLs
# =============================================================================
STORE_CACHE_TTL_SECONDS = 300  # 5 minutes

# =============================================================================
# API Configuration
# =============================================================================
DOCS_URL = "/docs"
REDOC_URL = "/redoc"
OPENAPI_URL = "/openapi.json"
DEFAULT_API_PREFIX = "/api/v1"

# =============================================================================
# Event Types (for audit logging)
# =============================================================================
EVENT_GENERATION_CACHE_HIT = "generation_cache_hit"
EVENT_GENERATION_STARTED = "generation_started"
EVENT_GENERATION_COMPLETED = "generation_completed"
EVENT_GENERATION_FAILED = "generation_failed"

# =============================================================================
# Generation Types
# =============================================================================
GENERATION_TYPE_SUMMARY = "summary"
GENERATION_TYPE_FAQS = "faqs"
GENERATION_TYPE_QUESTIONS = "questions"

# =============================================================================
# Search Modes (RAG)
# =============================================================================
SEARCH_MODE_SEMANTIC = "semantic"
SEARCH_MODE_KEYWORD = "keyword"
SEARCH_MODE_HYBRID = "hybrid"
VALID_SEARCH_MODES = [SEARCH_MODE_SEMANTIC, SEARCH_MODE_KEYWORD, SEARCH_MODE_HYBRID]


# =============================================================================
# Timeout Constants (consolidating scattered timeout values)
# =============================================================================
class Timeouts:
    """Centralized timeout values in seconds."""

    # LLM/Agent execution timeouts
    LLM_EXECUTION = 300  # 5 minutes - standard LLM call timeout
    QUESTION_GENERATION = 120  # 2 minutes - question generation
    SHEETS_REQUEST = 60  # 1 minute - sheets agent request

    # Queue/connection timeouts
    QUEUE_SHUTDOWN = 10  # Queue shutdown grace period
    CONNECTION_TEST = 15  # Database connection test
    PRE_WARM = 30.0  # Pre-warming timeout for DB connections

    # Retry-related timeouts
    RETRY_MIN_WAIT = 1  # Minimum wait between retries
    RETRY_MAX_WAIT = 32  # Maximum wait between retries (exponential backoff cap)
    RETRY_MAX_WAIT_DB = 10  # Database retry max wait

    # Bulk processing timeouts
    BULK_PARSE = 300  # Document parsing timeout
    BULK_GENERATION = 300  # Content generation timeout
    BULK_JOB = 3600  # Overall job timeout (1 hour)


# =============================================================================
# Retry Configuration Constants
# =============================================================================
class RetryConfig:
    """Centralized retry/backoff configuration."""

    MAX_ATTEMPTS = 3  # Default max retry attempts
    BACKOFF_FACTOR = 2.0  # Exponential backoff multiplier
    INITIAL_DELAY = 1.0  # Initial delay in seconds
    MIN_WAIT = 1  # Minimum wait between retries
    MAX_WAIT = 32  # Maximum wait (backoff cap)

    # Model-specific retries
    MODEL_MAX_ATTEMPTS = 3
    TOOL_MAX_ATTEMPTS = 2

    # Bulk processing retries
    BULK_MAX_RETRIES_PER_DOCUMENT = 3
    BULK_RETRY_DELAY = 30  # seconds


# =============================================================================
# Cache TTL Constants (consolidating scattered TTL values)
# =============================================================================
class CacheTTL:
    """Centralized cache time-to-live values in seconds."""

    # Short-lived caches
    JOB_STATUS = 3  # Bulk job status cache
    STATS = 30  # Statistics cache

    # Medium-lived caches
    QUOTA = 60  # Quota check cache (1 minute)
    STORE = 300  # File store cache (5 minutes)
    SCHEMA = 300  # Schema cache (5 minutes)

    # Long-lived caches
    SUBSCRIPTION_TIER = 3600  # Tier cache (1 hour)

    # Cache sizes
    SCHEMA_MAX_SIZE = 50  # Max cached schemas
    DOCUMENT_CACHE_SIZE = 1000  # LRU cache for documents


# =============================================================================
# Pagination Constants
# =============================================================================
class Pagination:
    """Centralized pagination defaults and limits."""

    DEFAULT_LIMIT = 20
    DEFAULT_OFFSET = 0

    # Standard limits
    MAX_LIMIT = 100

    # Large result sets
    LARGE_DEFAULT_LIMIT = 50
    LARGE_MAX_LIMIT = 200

    # Minimum
    MIN_LIMIT = 1


# =============================================================================
# PII Detection Constants
# =============================================================================
PII_STRATEGIES = frozenset(['redact', 'mask', 'hash', 'block'])
DEFAULT_PII_STRATEGY = 'redact'


# =============================================================================
# Queue Configuration
# =============================================================================
class QueueConfig:
    """Centralized queue configuration."""

    MAX_SIZE = 1000
    GET_TIMEOUT = 0.5
    SHUTDOWN_TIMEOUT = 5.0


# =============================================================================
# File Upload Configuration
# =============================================================================
class FileUpload:
    """Centralized file upload settings."""

    MAX_WORKERS = 3  # Concurrent file uploads
    WAIT_TIME = 0.5  # Upload retry wait time
    MAX_WAIT = 10  # Upload retry max wait
    SIGNED_URL_EXPIRATION_MINUTES = 60  # Signed URL expiration
