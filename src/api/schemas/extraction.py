"""API schemas for extraction endpoints."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Field Schemas
# =============================================================================

class DiscoveredFieldSchema(BaseModel):
    """A field discovered during document analysis."""
    field_name: str = Field(description="Machine-readable name (snake_case)")
    display_name: str = Field(description="Human-readable label")
    data_type: str = Field(description="Data type: string, number, date, currency, boolean, array")
    sample_value: Optional[str] = Field(default=None, description="Example value from document")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence 0.0-1.0")
    location: str = Field(description="Location in document: header, line_item, footer, body")
    required: bool = Field(default=False, description="Whether field appears mandatory")


class FieldSelectionSchema(BaseModel):
    """A field selected by the user for extraction."""
    field_name: str = Field(description="Machine-readable name")
    display_name: str = Field(description="Human-readable label")
    data_type: str = Field(description="Data type: string, number, date, currency, boolean")
    location: str = Field(description="Location: header, line_item, footer")
    required: bool = Field(default=False, description="Whether field is required")


# =============================================================================
# Analyze Fields Request/Response
# =============================================================================

class AnalyzeFieldsRequest(BaseModel):
    """Request to analyze document fields."""
    document_name: str = Field(
        ...,
        description="Name of the document",
        example="invoice_001.md"
    )
    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document",
        example="Acme corp/parsed/invoices/invoice_001.md"
    )
    document_type_hint: Optional[str] = Field(
        default=None,
        description="Hint about document type (invoice, contract, receipt, purchase_order)",
        example="invoice"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for tracking",
        example="sess_abc123"
    )

    @field_validator('parsed_file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Prevent path traversal attacks."""
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid path: path traversal not allowed")
        return v


class AnalyzeFieldsResponse(BaseModel):
    """Response from field analysis."""
    success: bool = Field(..., example=True)
    document_name: str = Field(..., example="invoice_001.md")
    document_type: Optional[str] = Field(default=None, example="invoice")
    fields: Optional[List[DiscoveredFieldSchema]] = Field(
        default=None,
        description="Discovered header/footer fields"
    )
    has_line_items: bool = Field(default=False, example=True)
    line_item_fields: Optional[List[DiscoveredFieldSchema]] = Field(
        default=None,
        description="Fields within line items"
    )
    processing_time_ms: float = Field(..., example=1234.56)
    error: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


# =============================================================================
# Generate Schema Request/Response
# =============================================================================

class GenerateSchemaRequest(BaseModel):
    """Request to generate extraction schema."""
    template_name: str = Field(
        ...,
        description="Name for the schema template",
        min_length=1,
        max_length=100,
        example="standard_invoice"
    )
    document_type: str = Field(
        ...,
        description="Type of document",
        example="invoice"
    )
    folder_name: str = Field(
        ...,
        description="Folder name for schema organization (e.g., 'invoices')",
        example="invoices"
    )
    selected_fields: List[FieldSelectionSchema] = Field(
        ...,
        description="Fields selected for extraction",
        min_length=1
    )
    save_template: bool = Field(
        default=True,
        description="Whether to save schema as a reusable template"
    )
    session_id: Optional[str] = Field(default=None)


class GenerateSchemaResponse(BaseModel):
    """Response from schema generation."""
    model_config = {"populate_by_name": True}

    success: bool = Field(..., example=True)
    template_name: str = Field(..., example="standard_invoice")
    document_type: Optional[str] = Field(default=None, example="invoice")
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generated JSON schema",
        serialization_alias="schema"
    )
    gcs_uri: Optional[str] = Field(
        default=None,
        description="GCS URI where schema was saved",
        example="gs://bucket/org/schemas/standard_invoice.json"
    )
    processing_time_ms: float = Field(..., example=234.56)
    error: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


# =============================================================================
# Extract Data Request/Response
# =============================================================================

class ExtractDataRequest(BaseModel):
    """Request to extract data from document."""
    model_config = {"populate_by_name": True}

    document_name: str = Field(
        ...,
        description="Name of the document",
        example="invoice_001.md"
    )
    parsed_file_path: str = Field(
        ...,
        description="GCS path to parsed document",
        example="Acme corp/parsed/invoices/invoice_001.md"
    )
    template_name: Optional[str] = Field(
        default=None,
        description="Name of saved template to use",
        example="standard_invoice"
    )
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for extraction (alternative to template_name)",
        alias="schema"
    )
    session_id: Optional[str] = Field(default=None)

    @field_validator('parsed_file_path')
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Prevent path traversal attacks."""
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid path: path traversal not allowed")
        return v


class TokenUsageSchema(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(default=0, example=500)
    completion_tokens: int = Field(default=0, example=200)
    total_tokens: int = Field(default=0, example=700)
    estimated_cost_usd: Optional[float] = Field(default=None, example=0.0007)


class ExtractDataResponse(BaseModel):
    """Response from data extraction."""
    success: bool = Field(..., example=True)
    extraction_job_id: Optional[str] = Field(
        default=None,
        description="Unique ID for this extraction job",
        example="ext_abc123"
    )
    document_name: str = Field(..., example="invoice_001.md")
    schema_title: Optional[str] = Field(default=None, example="standard_invoice")
    extracted_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted structured data"
    )
    extracted_field_count: int = Field(default=0, example=15)
    token_usage: Optional[TokenUsageSchema] = Field(default=None)
    cached: bool = Field(
        default=False,
        description="Whether result was retrieved from GCS cache"
    )
    processing_time_ms: float = Field(..., example=2345.67)
    error: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


# =============================================================================
# Template Management
# =============================================================================

class TemplateInfoSchema(BaseModel):
    """Information about an extraction template."""
    name: str = Field(..., example="standard_invoice")
    document_type: str = Field(..., example="invoice")
    created_at: Optional[str] = Field(default=None, example="2025-01-15T10:30:00Z")
    gcs_path: Optional[str] = Field(default=None)
    field_count: int = Field(default=0, example=12)


class TemplateListResponse(BaseModel):
    """Response listing extraction templates."""
    success: bool = Field(..., example=True)
    templates: List[TemplateInfoSchema] = Field(default_factory=list)
    total: int = Field(default=0, example=5)
    error: Optional[str] = Field(default=None)


class TemplateResponse(BaseModel):
    """Response with single template details."""
    model_config = {"populate_by_name": True}

    success: bool = Field(..., example=True)
    name: Optional[str] = Field(default=None)
    document_type: Optional[str] = Field(default=None)
    schema_definition: Optional[Dict[str, Any]] = Field(
        default=None,
        serialization_alias="schema"
    )
    gcs_path: Optional[str] = Field(default=None)
    error: Optional[str] = Field(default=None)


# =============================================================================
# Save Extracted Data
# =============================================================================

class SaveExtractedDataRequest(BaseModel):
    """Request to save extracted data to database."""
    model_config = {"populate_by_name": True}

    extraction_job_id: str = Field(
        ...,
        description="ID of the extraction job"
    )
    document_id: str = Field(
        ...,
        description="ID of the source document"
    )
    template_name: str = Field(
        ...,
        description="Name of the extraction template (required for table selection)",
        alias="template_id"
    )
    extracted_data: Dict[str, Any] = Field(
        ...,
        description="Extracted data to save"
    )
    source_file_path: Optional[str] = Field(
        default=None,
        description="Path to the source file"
    )
    folder_name: Optional[str] = Field(
        default=None,
        description="Folder name where the template is stored (e.g., 'invoices')"
    )


class SaveExtractedDataResponse(BaseModel):
    """Response from saving extracted data."""
    success: bool = Field(..., example=True)
    record_id: Optional[str] = Field(
        default=None,
        description="ID of the saved record"
    )
    table_name: Optional[str] = Field(
        default=None,
        description="Name of the table where data was saved"
    )
    message: str = Field(..., example="Data saved successfully")
    error: Optional[str] = Field(default=None)


class ExtractedRecordSummary(BaseModel):
    """Summary of an extracted record (without line items)."""
    id: str = Field(description="Record UUID")
    document_id: str = Field(description="Source document ID")
    extraction_job_id: str = Field(description="Extraction job ID")
    template_name: str = Field(description="Template used")
    extracted_at: Optional[str] = Field(default=None, description="Extraction timestamp")
    source_file_path: Optional[str] = Field(default=None)


class ExtractedRecordListResponse(BaseModel):
    """Response listing extracted records."""
    success: bool = Field(default=True)
    records: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = Field(default=0, description="Total count of records")
    limit: int = Field(default=100)
    offset: int = Field(default=0)
    error: Optional[str] = Field(default=None)


class ExtractedRecordDetailResponse(BaseModel):
    """Response with full record details including line items."""
    success: bool = Field(default=True)
    record: Optional[Dict[str, Any]] = Field(default=None)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)


# =============================================================================
# Error Responses
# =============================================================================

EXTRACTION_ERROR_RESPONSES = {
    400: {"description": "Invalid request parameters"},
    401: {"description": "API key required but not provided"},
    404: {"description": "Document or template not found"},
    429: {"description": "Rate limit exceeded"},
    500: {"description": "Internal server error"},
}
