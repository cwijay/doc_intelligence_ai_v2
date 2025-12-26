"""Pydantic schemas for Sheets Agent API."""

import os
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


class ChatRequest(BaseModel):
    """Request schema for sheets chat endpoint.

    Multi-tenancy: organization_id is used for tenant isolation.
    """

    file_paths: List[str] = Field(
        ...,
        description="List of local file paths to analyze",
        min_length=1,
        max_length=10
    )

    query: str = Field(
        ...,
        description="Natural language query about the data",
        min_length=1,
        max_length=2000
    )

    session_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation continuity"
    )

    user_id: Optional[str] = Field(
        default=None,
        description="User ID for long-term memory (preferences and history)"
    )

    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenant isolation"
    )

    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional options for processing"
    )

    @field_validator('file_paths')
    @classmethod
    def validate_paths(cls, paths: List[str]) -> List[str]:
        """Validate file paths to prevent path traversal attacks."""
        allowed_base = os.environ.get('ALLOWED_FILE_BASE', '/Users')
        for path in paths:
            # Resolve to absolute path
            real_path = os.path.realpath(path)
            # Check path stays within allowed directory
            if not real_path.startswith(allowed_base):
                raise ValueError(f"Path outside allowed directory: {path}")
            # Check for path traversal attempts
            if '..' in path:
                raise ValueError(f"Path traversal detected: {path}")
        return paths

    @field_validator('query')
    @classmethod
    def validate_query(cls, query: str) -> str:
        """Sanitize query input."""
        # Remove null bytes and control characters
        query = query.replace('\x00', '').strip()
        if not query:
            raise ValueError("Query cannot be empty")
        return query


class FileMetadata(BaseModel):
    """Metadata about processed files."""

    file_path: str = Field(..., description="File path")
    file_type: str = Field(..., description="File extension (.xlsx, .xls, .csv)")
    size_bytes: Optional[int] = Field(None, description="File size in bytes")
    shape: Optional[Dict[str, int]] = Field(None, description="Rows and columns count")
    columns: Optional[List[str]] = Field(None, description="Column names")
    processing_time_ms: float = Field(..., description="Time taken to process the file")


class ToolUsage(BaseModel):
    """Information about tools used during processing."""

    tool_name: str = Field(..., description="Name of the tool used")
    input_data: Dict[str, Any] = Field(..., description="Input provided to the tool")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output from the tool")
    execution_time_ms: float = Field(..., description="Tool execution time")
    success: bool = Field(..., description="Whether tool execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if tool failed")


class TokenUsage(BaseModel):
    """AI model token usage statistics."""

    prompt_tokens: int = Field(..., description="Tokens used in prompts")
    completion_tokens: int = Field(..., description="Tokens generated in completion")
    total_tokens: int = Field(..., description="Total tokens used")
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")


class ChatResponse(BaseModel):
    """Response schema for sheets chat endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Status message or error description")

    # Primary response data
    response: Optional[str] = Field(None, description="Natural language answer to the query")

    # File analysis metadata
    files_processed: List[FileMetadata] = Field(
        default_factory=list,
        description="Information about processed files"
    )

    # Tool usage tracking
    tools_used: List[ToolUsage] = Field(
        default_factory=list,
        description="Details about tools used during processing"
    )

    # Token consumption
    token_usage: Optional[TokenUsage] = Field(None, description="AI model usage statistics")

    # Session management
    session_id: str = Field(..., description="Session ID for conversation continuity")

    # Performance metrics
    total_processing_time_ms: float = Field(..., description="Total processing time")

    # Additional data (if any)
    data: Optional[Dict[str, Any]] = Field(None, description="Additional structured data")

    # Timestamp
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )


# Re-export SessionInfo from shared module for backward compatibility
from src.agents.core.session_manager import SessionInfo


class ErrorResponse(BaseModel):
    """Error response schema."""

    success: bool = Field(False, description="Always false for errors")
    error_type: str = Field(..., description="Category of error")
    message: str = Field(..., description="User-friendly error message")
    detail: Optional[str] = Field(None, description="Technical error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

    # Request context
    session_id: Optional[str] = Field(None, description="Session ID if available")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")

    # Retry guidance
    retryable: bool = Field(False, description="Whether the request can be retried")
    retry_after_seconds: Optional[int] = Field(None, description="Suggested retry delay")
