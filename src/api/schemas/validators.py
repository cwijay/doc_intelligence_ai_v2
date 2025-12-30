"""Shared validators for Pydantic schemas.

This module provides reusable validation functions that can be used
across different schema definitions to eliminate code duplication.
"""

from typing import Optional

from src.constants import (
    MIN_NUM_FAQS,
    MAX_NUM_FAQS,
    MIN_NUM_QUESTIONS,
    MAX_NUM_QUESTIONS,
    MIN_SUMMARY_WORDS,
    MAX_SUMMARY_WORDS,
    VALID_SEARCH_MODES,
)


def validate_parsed_file_path(v: str) -> str:
    """Validate parsed_file_path to prevent path traversal attacks.

    Args:
        v: The file path to validate.

    Returns:
        The validated and stripped path.

    Raises:
        ValueError: If the path contains path traversal patterns.
    """
    if '..' in v:
        raise ValueError('Invalid path: ".." not allowed')
    if v.startswith('/'):
        raise ValueError('Invalid path: must be relative, not absolute')
    if '\\' in v:
        raise ValueError('Invalid path: backslashes not allowed')
    return v.strip()


def validate_document_name(v: str) -> str:
    """Validate document name to prevent path traversal.

    Args:
        v: The document name to validate.

    Returns:
        The validated and stripped document name.

    Raises:
        ValueError: If the name contains path traversal patterns.
    """
    if '..' in v or '/' in v or '\\' in v:
        raise ValueError("Invalid document name: path traversal not allowed")
    return v.strip()


def validate_query(query: str) -> str:
    """Sanitize and validate query input.

    Args:
        query: The query string to validate.

    Returns:
        The sanitized query string.

    Raises:
        ValueError: If the query is empty after sanitization.
    """
    query = query.replace('\x00', '').strip()
    if not query:
        raise ValueError("Query cannot be empty")
    return query


def validate_num_faqs(v: Optional[int]) -> Optional[int]:
    """Validate number of FAQs.

    Args:
        v: The number of FAQs to validate (can be None).

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is out of range.
    """
    if v is not None and not MIN_NUM_FAQS <= v <= MAX_NUM_FAQS:
        raise ValueError(f"num_faqs must be between {MIN_NUM_FAQS} and {MAX_NUM_FAQS}")
    return v


def validate_num_questions(v: Optional[int]) -> Optional[int]:
    """Validate number of questions.

    Args:
        v: The number of questions to validate (can be None).

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is out of range.
    """
    if v is not None and not MIN_NUM_QUESTIONS <= v <= MAX_NUM_QUESTIONS:
        raise ValueError(f"num_questions must be between {MIN_NUM_QUESTIONS} and {MAX_NUM_QUESTIONS}")
    return v


def validate_summary_max_words(v: Optional[int]) -> Optional[int]:
    """Validate summary max words.

    Args:
        v: The max words to validate (can be None).

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is out of range.
    """
    if v is not None and not MIN_SUMMARY_WORDS <= v <= MAX_SUMMARY_WORDS:
        raise ValueError(f"summary_max_words must be between {MIN_SUMMARY_WORDS} and {MAX_SUMMARY_WORDS}")
    return v


def validate_search_mode(v: str) -> str:
    """Validate search mode.

    Args:
        v: The search mode to validate.

    Returns:
        The validated and lowercased search mode.

    Raises:
        ValueError: If the search mode is not valid.
    """
    if v.lower() not in VALID_SEARCH_MODES:
        raise ValueError(f"search_mode must be one of: {VALID_SEARCH_MODES}")
    return v.lower()


def validate_top_k(v: int, min_val: int = 1, max_val: int = 20) -> int:
    """Validate top_k parameter for search queries.

    Args:
        v: The top_k value to validate.
        min_val: Minimum allowed value (default 1).
        max_val: Maximum allowed value (default 20).

    Returns:
        The validated value.

    Raises:
        ValueError: If the value is out of range.
    """
    if not min_val <= v <= max_val:
        raise ValueError(f"top_k must be between {min_val} and {max_val}")
    return v
