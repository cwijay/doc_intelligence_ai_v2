"""Document Agent API schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, field_validator

from .common import TokenUsage, DifficultyLevel


def _validate_parsed_file_path(v: str) -> str:
    """Validate parsed_file_path to prevent path traversal attacks."""
    if '..' in v:
        raise ValueError('Invalid path: ".." not allowed')
    if v.startswith('/'):
        raise ValueError('Invalid path: must be relative, not absolute')
    if '\\' in v:
        raise ValueError('Invalid path: backslashes not allowed')
    return v


# =============================================================================
# Request Models
# =============================================================================

class GenerationOptions(BaseModel):
    """Options for content generation."""
    num_faqs: int = Field(default=10, ge=1, le=50, example=10)
    num_questions: int = Field(default=10, ge=1, le=100, example=10)
    summary_max_words: int = Field(default=500, ge=50, le=2000, example=500)


class DocumentProcessRequest(BaseModel):
    """Request for full document processing."""
    document_name: str = Field(..., description="Name of the document to process", example="Sample1.md")
    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')",
        example="Acme corp/parsed/invoices/Sample1.md"
    )
    query: str = Field(..., description="Processing query/instruction", example="Generate a summary of this document")
    session_id: Optional[str] = Field(None, description="Session ID for context", example="sess_abc123")
    user_id: Optional[str] = Field(None, description="User ID for personalization", example="user_123")
    options: Optional[GenerationOptions] = None

    @field_validator('parsed_file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        return _validate_parsed_file_path(v)


class SummarizeRequest(BaseModel):
    """Request for summary generation."""
    document_name: str = Field(..., description="Name of the document", example="Sample1.md")
    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')",
        example="Acme corp/parsed/invoices/Sample1.md"
    )
    max_words: int = Field(default=500, ge=50, le=2000, example=300)
    session_id: Optional[str] = Field(default=None, example="sess_abc123")
    force: bool = Field(default=False, description="Force regeneration, bypassing GCS cache")

    @field_validator('parsed_file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        return _validate_parsed_file_path(v)


class FAQsRequest(BaseModel):
    """Request for FAQ generation."""
    document_name: str = Field(..., description="Name of the document", example="Sample1.md")
    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')",
        example="Acme corp/parsed/invoices/Sample1.md"
    )
    num_faqs: int = Field(default=10, ge=1, le=50, example=10)
    session_id: Optional[str] = Field(default=None, example="sess_abc123")
    force: bool = Field(default=False, description="Force regeneration, bypassing GCS cache")

    @field_validator('parsed_file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        return _validate_parsed_file_path(v)


class QuestionsRequest(BaseModel):
    """Request for question generation."""
    document_name: str = Field(..., description="Name of the document", example="Sample1.md")
    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')",
        example="Acme corp/parsed/invoices/Sample1.md"
    )
    num_questions: int = Field(default=10, ge=1, le=100, example=10)
    session_id: Optional[str] = Field(default=None, example="sess_abc123")
    force: bool = Field(default=False, description="Force regeneration, bypassing GCS cache")

    @field_validator('parsed_file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        return _validate_parsed_file_path(v)


class GenerateAllRequest(BaseModel):
    """Request for generating all content types."""
    document_name: str = Field(..., description="Name of the document", example="Sample1.md")
    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document (e.g., 'Acme corp/parsed/invoices/Sample1.md')",
        example="Acme corp/parsed/invoices/Sample1.md"
    )
    options: Optional[GenerationOptions] = None
    session_id: Optional[str] = Field(default=None, example="sess_abc123")
    force: bool = Field(default=False, description="Force regeneration, bypassing GCS cache")

    @field_validator('parsed_file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        return _validate_parsed_file_path(v)


# =============================================================================
# Response Models
# =============================================================================

class FAQ(BaseModel):
    """A single FAQ item."""
    question: str = Field(..., example="What is the main topic of this document?")
    answer: str = Field(..., example="The document discusses AI-powered document analysis techniques.")


class Question(BaseModel):
    """A single comprehension question."""
    question: str = Field(..., example="What are the key benefits mentioned in the document?")
    expected_answer: str = Field(..., example="The document highlights improved efficiency and accuracy.")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.medium, description="Question difficulty level", example="medium")


class GeneratedContent(BaseModel):
    """Generated content container."""
    summary: Optional[str] = None
    faqs: Optional[List[FAQ]] = None
    questions: Optional[List[Question]] = None


class DocumentMetadata(BaseModel):
    """Document metadata."""
    document_name: str = Field(..., example="Sample1.md")
    org_name: Optional[str] = Field(default=None, description="Organization name", example="ACME corp")
    folder_name: Optional[str] = Field(default=None, description="Folder name", example="invoices")
    source_path: Optional[str] = Field(default=None, example="ACME corp/parsed/invoices/Sample1.md")
    source_type: Optional[str] = Field(default=None, example="parsed")
    file_size_bytes: Optional[int] = Field(default=None, example=15234)
    content_length: Optional[int] = Field(default=None, example=12500)


class DocumentProcessResponse(BaseModel):
    """Response for document processing."""
    success: bool = Field(..., example=True)
    message: Optional[str] = Field(default=None, example="Document processed successfully")
    document_name: str = Field(..., example="Sample1.md")
    content: Optional[GeneratedContent] = None
    metadata: Optional[DocumentMetadata] = None
    token_usage: Optional[TokenUsage] = None
    processing_time_ms: float = Field(..., example=1523.45)
    session_id: Optional[str] = Field(default=None, example="sess_abc123")
    cached: bool = Field(default=False, example=False)


class SummarizeResponse(BaseModel):
    """Response for summary generation."""
    success: bool = Field(..., example=True)
    summary: Optional[str] = Field(default=None, example="This document provides an overview of...")
    word_count: int = Field(default=0, example=245)
    cached: bool = Field(default=False, example=False)
    processing_time_ms: float = Field(..., example=892.34)
    error: Optional[str] = None


class FAQsResponse(BaseModel):
    """Response for FAQ generation."""
    success: bool = Field(..., example=True)
    faqs: Optional[List[FAQ]] = None
    count: int = Field(default=0, example=5)
    cached: bool = Field(default=False, example=False)
    processing_time_ms: float = Field(..., example=1245.67)
    error: Optional[str] = None


class QuestionsResponse(BaseModel):
    """Response for question generation."""
    success: bool = Field(..., example=True)
    questions: Optional[List[Question]] = None
    count: int = Field(default=0, example=10)
    difficulty_distribution: Optional[Dict[str, int]] = Field(default=None, example={"easy": 3, "medium": 4, "hard": 3})
    cached: bool = Field(default=False, example=False)
    processing_time_ms: float = Field(..., example=1567.89)
    error: Optional[str] = None


class GenerateAllResponse(BaseModel):
    """Response for generating all content types."""
    success: bool = Field(..., example=True)
    document_name: str = Field(..., example="Sample1.md")
    summary: Optional[str] = Field(default=None, example="This document provides an overview of...")
    faqs: Optional[List[FAQ]] = None
    questions: Optional[List[Question]] = None
    cached: bool = Field(default=False, example=False)
    processing_time_ms: float = Field(..., example=3245.67)
    error: Optional[str] = None


# =============================================================================
# RAG Chat Models
# =============================================================================

class RAGChatRequest(BaseModel):
    """Request for conversational RAG chat."""
    query: str = Field(
        ...,
        description="User's question or search query",
        min_length=1,
        max_length=2000,
        example="What are the payment terms in the contract?"
    )
    organization_name: str = Field(
        ...,
        description="Organization name for store lookup",
        example="Acme Corp"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity",
        example="sess_abc123"
    )
    folder_filter: Optional[str] = Field(
        default=None,
        description="Filter search to specific folder",
        example="Legal"
    )
    file_filter: Optional[str] = Field(
        default=None,
        description="Filter search to specific file",
        example="contract.pdf"
    )
    search_mode: str = Field(
        default="hybrid",
        description="Search mode: 'semantic', 'keyword', or 'hybrid'",
        example="hybrid"
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of citations to return",
        example=5
    )

    @field_validator('search_mode')
    @classmethod
    def validate_search_mode(cls, v: str) -> str:
        valid_modes = ['semantic', 'keyword', 'hybrid']
        if v.lower() not in valid_modes:
            raise ValueError(f"search_mode must be one of: {valid_modes}")
        return v.lower()


class RAGCitation(BaseModel):
    """A citation from RAG search results."""
    text: str = Field(
        ...,
        description="Cited text from the source document",
        example="Payment shall be due within 30 days of invoice date."
    )
    file: str = Field(
        ...,
        description="Source file name",
        example="contract.md"
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score (0-1)",
        example=0.92
    )
    folder_name: Optional[str] = Field(
        default=None,
        description="Folder containing the file",
        example="Legal"
    )
    page: Optional[int] = Field(
        default=None,
        description="Page number if applicable",
        example=3
    )


class RAGChatResponse(BaseModel):
    """Response for conversational RAG chat."""
    success: bool = Field(..., example=True)
    answer: str = Field(
        ...,
        description="Generated answer from RAG",
        example="Based on the contract, payment terms are Net 30 days from invoice date."
    )
    citations: List[RAGCitation] = Field(
        default_factory=list,
        description="Source citations for the answer"
    )
    query: str = Field(..., description="Original query", example="What are the payment terms?")
    search_mode: str = Field(..., description="Search mode used", example="hybrid")
    filters: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Applied filters (folder, file)",
        example={"folder": "Legal", "file": None}
    )
    session_id: str = Field(..., description="Session ID for conversation continuity", example="sess_abc123")
    processing_time_ms: float = Field(..., description="Total processing time", example=1245.67)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )
    error: Optional[str] = Field(default=None, description="Error message if search failed")
