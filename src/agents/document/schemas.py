"""Pydantic schemas for Document Agent API."""

import os
import uuid
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.api.schemas.validators import (
    validate_parsed_file_path,
    validate_document_name,
    validate_query as _validate_query,
    validate_num_faqs,
    validate_num_questions,
    validate_summary_max_words,
)


class GenerationOptions(BaseModel):
    """Options for content generation."""

    num_faqs: Optional[int] = Field(
        None,
        description="Number of FAQs to generate (overrides default)"
    )

    num_questions: Optional[int] = Field(
        None,
        description="Number of questions to generate (overrides default)"
    )

    summary_max_words: Optional[int] = Field(
        None,
        description="Maximum words for summary (overrides default)"
    )

    @field_validator('num_faqs')
    @classmethod
    def _validate_num_faqs(cls, v: Optional[int]) -> Optional[int]:
        return validate_num_faqs(v)

    @field_validator('num_questions')
    @classmethod
    def _validate_num_questions(cls, v: Optional[int]) -> Optional[int]:
        return validate_num_questions(v)

    @field_validator('summary_max_words')
    @classmethod
    def _validate_summary_words(cls, v: Optional[int]) -> Optional[int]:
        return validate_summary_max_words(v)


class DocumentRequest(BaseModel):
    """Request schema for document processing endpoint.

    Multi-tenancy: organization_id is used for tenant isolation.
    """

    document_name: str = Field(
        ...,
        description="Name of the document to process (e.g., 'Sample1.md' or 'report.txt')",
        min_length=1,
        max_length=255
    )

    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')",
        min_length=1,
        max_length=1024
    )

    query: str = Field(
        ...,
        description="Natural language query about the document",
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

    file_filter: Optional[str] = Field(
        default=None,
        description="File name filter for RAG cache scoping (single document context)"
    )

    folder_filter: Optional[str] = Field(
        default=None,
        description="Folder name filter for RAG cache scoping (folder context)"
    )

    options: Optional[GenerationOptions] = Field(
        default=None,
        description="Optional generation settings"
    )

    @field_validator('document_name')
    @classmethod
    def _validate_document_name(cls, v: str) -> str:
        """Validate document name to prevent path traversal."""
        return validate_document_name(v)

    @field_validator('parsed_file_path')
    @classmethod
    def _validate_parsed_file_path(cls, v: str) -> str:
        """Validate parsed_file_path to prevent path traversal."""
        return validate_parsed_file_path(v)

    @field_validator('query')
    @classmethod
    def validate_query(cls, query: str) -> str:
        """Sanitize query input."""
        return _validate_query(query)


class FAQ(BaseModel):
    """A single FAQ pair."""

    question: str = Field(..., description="The question")
    answer: str = Field(..., description="The answer")


class Question(BaseModel):
    """A comprehension question."""

    question: str = Field(..., description="The question text")
    expected_answer: Optional[str] = Field(
        None,
        description="Expected or sample answer"
    )
    difficulty: Optional[str] = Field(
        None,
        description="Difficulty level: easy, medium, hard"
    )

    @field_validator('difficulty')
    @classmethod
    def validate_difficulty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in ['easy', 'medium', 'hard']:
            raise ValueError("difficulty must be 'easy', 'medium', or 'hard'")
        return v.lower() if v else None


class GeneratedContent(BaseModel):
    """Container for all generated content."""

    summary: Optional[str] = Field(None, description="Document summary")
    faqs: Optional[List[FAQ]] = Field(None, description="Generated FAQs")
    questions: Optional[List[Question]] = Field(None, description="Generated questions")


class TokenUsage(BaseModel):
    """AI model token usage statistics."""

    prompt_tokens: int = Field(..., description="Tokens used in prompts")
    completion_tokens: int = Field(..., description="Tokens generated in completion")
    total_tokens: int = Field(..., description="Total tokens used")
    estimated_cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")


class DocumentMetadata(BaseModel):
    """Metadata about processed document."""

    document_name: str = Field(..., description="Document name")
    parsed_file_path: Optional[str] = Field(None, description="GCS path to parsed document")
    source_path: str = Field(..., description="Full path where document was found")
    source_type: str = Field(..., description="Source: 'parsed_gcs' or 'upload'")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    content_length: Optional[int] = Field(None, description="Content character count")


class ToolUsage(BaseModel):
    """Information about tools used during processing."""

    tool_name: str = Field(..., description="Name of the tool used")
    input_data: Dict[str, Any] = Field(..., description="Input provided to the tool")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output from the tool")
    execution_time_ms: float = Field(..., description="Tool execution time")
    success: bool = Field(..., description="Whether tool execution was successful")
    error_message: Optional[str] = Field(None, description="Error message if tool failed")


class DocumentResponse(BaseModel):
    """Response schema for document processing endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Status message or error description")

    # Agent response text (for RAG chat answers)
    response_text: Optional[str] = Field(
        None,
        description="The agent's response text (used for RAG chat answers)"
    )

    # Document info
    document_name: str = Field(..., description="Processed document name")
    source_path: Optional[str] = Field(None, description="Path where document was found")

    # Generated content
    content: Optional[GeneratedContent] = Field(
        None,
        description="Generated summary, FAQs, and questions"
    )

    # Processing metadata
    document_metadata: Optional[DocumentMetadata] = Field(
        None,
        description="Document metadata"
    )

    # Tool usage tracking
    tools_used: List[ToolUsage] = Field(
        default_factory=list,
        description="Details about tools used during processing"
    )

    # Token consumption
    token_usage: Optional[TokenUsage] = Field(
        None,
        description="AI model usage statistics"
    )

    # Session management
    session_id: str = Field(..., description="Session ID for conversation continuity")

    # Performance metrics
    processing_time_ms: float = Field(..., description="Total processing time")

    # Persistence info
    persisted: bool = Field(False, description="Whether content was saved to database")
    database_id: Optional[str] = Field(None, description="Database record ID if persisted")
    output_file_paths: Optional[Dict[str, str]] = Field(
        None,
        description="GCS paths per content type: {summary: ..., faqs: ..., questions: ...}"
    )

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
    document_name: Optional[str] = Field(None, description="Document name if available")

    # Retry guidance
    retryable: bool = Field(False, description="Whether the request can be retried")
    retry_after_seconds: Optional[int] = Field(None, description="Suggested retry delay")


# =============================================================================
# RAG Chat Schemas
# =============================================================================

class RAGCitation(BaseModel):
    """A citation from RAG search results."""

    text: str = Field(..., description="Cited text from the source document")
    file: str = Field(..., description="Source file name")
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)"
    )
    folder_name: Optional[str] = Field(None, description="Folder containing the file")
    page: Optional[int] = Field(None, description="Page number if applicable")


class RAGChatRequest(BaseModel):
    """Request schema for conversational RAG chat."""

    query: str = Field(
        ...,
        description="User's question or search query",
        min_length=1,
        max_length=2000
    )

    session_id: Optional[str] = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Session ID for conversation continuity"
    )

    folder_filter: Optional[str] = Field(
        None,
        description="Filter search to specific folder"
    )

    file_filter: Optional[str] = Field(
        None,
        description="Filter search to specific file"
    )

    search_mode: str = Field(
        default="hybrid",
        description="Search mode: 'semantic', 'keyword', or 'hybrid'"
    )

    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of citations to return"
    )

    @field_validator('query')
    @classmethod
    def validate_query(cls, query: str) -> str:
        """Sanitize query input."""
        query = query.replace('\x00', '').strip()
        if not query:
            raise ValueError("Query cannot be empty")
        return query

    @field_validator('search_mode')
    @classmethod
    def validate_search_mode(cls, v: str) -> str:
        """Validate search mode."""
        valid_modes = ['semantic', 'keyword', 'hybrid']
        if v.lower() not in valid_modes:
            raise ValueError(f"search_mode must be one of: {valid_modes}")
        return v.lower()


class RAGChatResponse(BaseModel):
    """Response schema for conversational RAG chat."""

    success: bool = Field(..., description="Whether the search was successful")
    answer: str = Field(..., description="Generated answer from RAG")
    citations: List[RAGCitation] = Field(
        default_factory=list,
        description="Source citations for the answer"
    )

    # Search metadata
    query: str = Field(..., description="Original query")
    search_mode: str = Field(..., description="Search mode used")
    filters: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Applied filters (folder, file)"
    )

    # Session management
    session_id: str = Field(..., description="Session ID for conversation continuity")

    # Performance metrics
    processing_time_ms: float = Field(..., description="Total processing time")

    # Timestamp
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )

    # Error info (if success=False)
    error: Optional[str] = Field(None, description="Error message if search failed")
