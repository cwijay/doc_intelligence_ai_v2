"""Shared error response definitions for OpenAPI documentation.

This module consolidates error response dictionaries that were previously
duplicated across multiple router files. Use these in FastAPI route
definitions for consistent error documentation.
"""

from .common import ErrorResponse

# =============================================================================
# Base Error Responses
# =============================================================================

BASE_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request parameters"},
    401: {"model": ErrorResponse, "description": "API key required but not provided"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}

# =============================================================================
# API-Specific Error Responses
# =============================================================================

DOCUMENT_ERROR_RESPONSES = {
    **BASE_ERROR_RESPONSES,
    403: {"model": ErrorResponse, "description": "Invalid API key or insufficient permissions"},
    404: {"model": ErrorResponse, "description": "Document not found"},
}

SHEETS_ERROR_RESPONSES = {
    **BASE_ERROR_RESPONSES,
    403: {"model": ErrorResponse, "description": "Invalid API key or insufficient permissions"},
    404: {"model": ErrorResponse, "description": "File not found"},
}

FILE_ERROR_RESPONSES = {
    **BASE_ERROR_RESPONSES,
    403: {"model": ErrorResponse, "description": "Invalid API key or insufficient permissions"},
    404: {"model": ErrorResponse, "description": "File not found"},
    413: {"model": ErrorResponse, "description": "File too large"},
}

STORE_ERROR_RESPONSES = {
    **BASE_ERROR_RESPONSES,
    403: {"model": ErrorResponse, "description": "Store belongs to another organization"},
    404: {"model": ErrorResponse, "description": "Store not found"},
}

FOLDER_ERROR_RESPONSES = {
    **BASE_ERROR_RESPONSES,
    403: {"model": ErrorResponse, "description": "Folder belongs to another organization"},
    404: {"model": ErrorResponse, "description": "Folder not found"},
}

AUDIT_ERROR_RESPONSES = {
    **BASE_ERROR_RESPONSES,
    403: {"model": ErrorResponse, "description": "Invalid API key or insufficient permissions"},
    404: {"model": ErrorResponse, "description": "Resource not found"},
}

SESSION_ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "API key required but not provided"},
    403: {"model": ErrorResponse, "description": "Session belongs to another organization"},
    404: {"model": ErrorResponse, "description": "Session not found"},
    500: {"model": ErrorResponse, "description": "Internal server error"},
}
