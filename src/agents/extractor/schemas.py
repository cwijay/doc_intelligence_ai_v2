"""Pydantic schemas for Extractor Agent requests and responses."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# Token Usage
# =============================================================================

class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(default=0, description="Tokens used in prompt")
    completion_tokens: int = Field(default=0, description="Tokens in response")
    total_tokens: int = Field(default=0, description="Total tokens used")
    estimated_cost_usd: Optional[float] = Field(default=None, description="Estimated cost")


# =============================================================================
# Discovered Fields
# =============================================================================

class DiscoveredField(BaseModel):
    """A field discovered during document analysis."""
    field_name: str = Field(description="Machine-readable name (snake_case)")
    display_name: str = Field(description="Human-readable label")
    data_type: str = Field(description="Data type: string, number, date, currency, boolean, array")
    sample_value: Optional[str] = Field(default=None, description="Example value from document")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0")
    location: str = Field(description="Location in document: header, line_item, footer, body")
    required: bool = Field(default=False, description="Whether field appears mandatory")


class FieldSelection(BaseModel):
    """A field selected by the user for extraction."""
    field_name: str = Field(description="Machine-readable name")
    display_name: str = Field(description="Human-readable label")
    data_type: str = Field(description="Data type")
    location: str = Field(description="Location: header, line_item, footer")
    required: bool = Field(default=False, description="Whether field is required")


# =============================================================================
# Analyze Fields
# =============================================================================

class AnalyzeFieldsRequest(BaseModel):
    """Request to analyze document fields."""
    content: str = Field(description="Parsed document content")
    document_name: str = Field(description="Name of the document")
    document_type_hint: Optional[str] = Field(
        default=None,
        description="Hint about document type (invoice, contract, receipt, etc.)"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenancy"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking"
    )


class AnalyzeFieldsResponse(BaseModel):
    """Response from field analysis."""
    success: bool = Field(description="Whether analysis succeeded")
    document_name: str = Field(description="Name of the document analyzed")
    document_type: Optional[str] = Field(
        default=None,
        description="Detected document type"
    )
    fields: Optional[List[DiscoveredField]] = Field(
        default=None,
        description="Discovered header/footer fields"
    )
    has_line_items: bool = Field(
        default=False,
        description="Whether document contains line items"
    )
    line_item_fields: Optional[List[DiscoveredField]] = Field(
        default=None,
        description="Fields within line items"
    )
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Token usage for this request")


# =============================================================================
# Generate Schema
# =============================================================================

class GenerateSchemaRequest(BaseModel):
    """Request to generate extraction schema."""
    selected_fields: List[FieldSelection] = Field(
        description="Fields selected for extraction"
    )
    template_name: str = Field(
        description="Name for the schema template",
        min_length=1,
        max_length=100
    )
    document_type: str = Field(
        description="Type of document (invoice, contract, receipt, etc.)"
    )
    organization_id: str = Field(description="Organization ID for scoping")
    save_to_gcs: bool = Field(
        default=True,
        description="Whether to save schema to GCS as template"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking"
    )


class GenerateSchemaResponse(BaseModel):
    """Response from schema generation."""
    success: bool = Field(description="Whether generation succeeded")
    template_name: str = Field(description="Name of the template")
    document_type: Optional[str] = Field(
        default=None,
        description="Document type"
    )
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generated JSON schema"
    )
    gcs_uri: Optional[str] = Field(
        default=None,
        description="GCS URI where schema was saved"
    )
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Token usage for this request")


# =============================================================================
# Extract Data
# =============================================================================

class ExtractDataRequest(BaseModel):
    """Request to extract data from document."""
    content: str = Field(description="Parsed document content")
    document_name: str = Field(description="Name of the document")
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for extraction (provide this OR template_name)"
    )
    template_name: Optional[str] = Field(
        default=None,
        description="Name of saved template to use (provide this OR schema)"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenancy"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking"
    )


class ExtractDataResponse(BaseModel):
    """Response from data extraction."""
    success: bool = Field(description="Whether extraction succeeded")
    extraction_job_id: Optional[str] = Field(
        default=None,
        description="Unique ID for this extraction job"
    )
    document_name: str = Field(description="Name of the document")
    schema_title: Optional[str] = Field(
        default=None,
        description="Title of the schema used"
    )
    extracted_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted structured data"
    )
    extracted_field_count: int = Field(
        default=0,
        description="Number of fields extracted"
    )
    token_usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage for extraction"
    )
    cached: bool = Field(
        default=False,
        description="Whether result was retrieved from cache"
    )
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    session_id: Optional[str] = Field(default=None, description="Session ID")


# =============================================================================
# Template Management
# =============================================================================

class TemplateInfo(BaseModel):
    """Information about an extraction template."""
    name: str = Field(description="Template name")
    document_type: str = Field(description="Document type")
    created_at: Optional[str] = Field(default=None, description="Creation timestamp")
    gcs_path: Optional[str] = Field(default=None, description="GCS storage path")
    field_count: int = Field(default=0, description="Number of fields in schema")


class TemplateListResponse(BaseModel):
    """Response listing extraction templates."""
    success: bool = Field(description="Whether listing succeeded")
    templates: List[TemplateInfo] = Field(
        default_factory=list,
        description="List of templates"
    )
    total: int = Field(default=0, description="Total number of templates")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class TemplateResponse(BaseModel):
    """Response with single template details."""
    success: bool = Field(description="Whether retrieval succeeded")
    template: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Template schema"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


# =============================================================================
# Full Extraction Request (combines all steps)
# =============================================================================

class ExtractionRequest(BaseModel):
    """Full extraction request (for process_request compatibility)."""
    document_name: str = Field(description="Name of the document")
    parsed_file_path: str = Field(description="GCS path to parsed document")
    action: str = Field(
        description="Action to perform: analyze, generate_schema, extract"
    )
    content: Optional[str] = Field(
        default=None,
        description="Document content (if not loading from path)"
    )
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Schema for extraction"
    )
    template_name: Optional[str] = Field(
        default=None,
        description="Template name for schema"
    )
    selected_fields: Optional[List[FieldSelection]] = Field(
        default=None,
        description="Fields for schema generation"
    )
    document_type: Optional[str] = Field(
        default=None,
        description="Document type hint"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID"
    )


class ExtractionResponse(BaseModel):
    """Full extraction response."""
    success: bool = Field(description="Whether request succeeded")
    action: str = Field(description="Action performed")
    message: str = Field(description="Result message")
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Result data (depends on action)"
    )
    processing_time_ms: float = Field(description="Processing time")
    error: Optional[str] = Field(default=None, description="Error if failed")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
