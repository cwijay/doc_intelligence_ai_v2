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
