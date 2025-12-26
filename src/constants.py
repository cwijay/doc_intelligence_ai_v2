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
