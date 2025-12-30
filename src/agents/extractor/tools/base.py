"""Shared utilities and input schemas for extractor tools.

This module contains common functions, formatters, and Pydantic schemas
used across the extraction tools (field analyzer, schema generator, data extractor).
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Type

from pydantic import BaseModel, Field, create_model

logger = logging.getLogger(__name__)


# =============================================================================
# Path Derivation Utilities
# =============================================================================

def derive_org_and_folder(parsed_file_path: str) -> tuple:
    """
    Extract org_name and folder_name from parsed_file_path.

    Example: "Acme corp/parsed/invoices/Sample1.md" -> ("Acme corp", "invoices")
    """
    if not parsed_file_path or 'parsed' not in parsed_file_path:
        return "", ""

    parts = parsed_file_path.split('/')
    try:
        parsed_idx = parts.index('parsed')
        org_name = '/'.join(parts[:parsed_idx])
        folder_parts = parts[parsed_idx + 1:-1]  # Everything between 'parsed' and filename
        folder_name = '/'.join(folder_parts) if folder_parts else ""
        return org_name, folder_name
    except ValueError:
        return "", ""


def derive_document_base(document_name: str) -> str:
    """
    Get document name without extension.

    Example: "Sample1.md" -> "Sample1"
    """
    return Path(document_name).stem


def build_schema_path(organization_id: str, template_name: str) -> str:
    """
    Build GCS path for extraction schema template.

    Args:
        organization_id: Organization ID
        template_name: Template name

    Returns:
        GCS path, e.g., "org_123/schemas/invoice_template.json"
    """
    safe_name = template_name.replace(' ', '_').lower()
    return f"{organization_id}/schemas/{safe_name}.json"


def build_extracted_path(
    organization_id: str,
    folder_name: str,
    document_name: str
) -> str:
    """
    Build GCS path for extracted data Excel export.

    Args:
        organization_id: Organization ID
        folder_name: Folder containing the document
        document_name: Original document name

    Returns:
        GCS path, e.g., "org_123/extracted/invoices/sample1_extracted.xlsx"
    """
    doc_base = derive_document_base(document_name)
    parts = [organization_id, "extracted"]
    if folder_name:
        parts.append(folder_name)
    parts.append(f"{doc_base}_extracted.xlsx")
    return '/'.join(parts)


# =============================================================================
# JSON Schema Utilities
# =============================================================================

JSON_TYPE_MAP = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "date": str,  # Dates as ISO strings
    "currency": float,
    "array": list,
    "object": dict,
}

PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def json_type_to_python(json_type: str) -> type:
    """Convert JSON schema type to Python type."""
    return JSON_TYPE_MAP.get(json_type.lower(), str)


def python_type_to_json(python_type: type) -> str:
    """Convert Python type to JSON schema type."""
    return PYTHON_TO_JSON_TYPE.get(python_type, "string")


def schema_to_pydantic(schema: Dict[str, Any], model_name: str = "DynamicExtraction") -> Type[BaseModel]:
    """
    Dynamically create a Pydantic model from a JSON schema.

    Args:
        schema: JSON schema dict with 'properties' and 'required' fields
        model_name: Name for the generated model

    Returns:
        Dynamically created Pydantic model class
    """
    fields = {}
    required_fields = set(schema.get("required", []))
    properties = schema.get("properties", {})

    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "string")
        description = field_def.get("description", "")

        # Handle array types (for line items)
        if field_type == "array":
            items = field_def.get("items", {})
            if items.get("type") == "object":
                # Recursively create nested model for array items
                nested_model = schema_to_pydantic(
                    {"properties": items.get("properties", {}), "required": items.get("required", [])},
                    f"{model_name}_{field_name}_item"
                )
                python_type = List[nested_model]
            else:
                item_type = json_type_to_python(items.get("type", "string"))
                python_type = List[item_type]
        elif field_type == "object":
            # Recursively create nested model
            nested_schema = {
                "properties": field_def.get("properties", {}),
                "required": field_def.get("required", [])
            }
            python_type = schema_to_pydantic(nested_schema, f"{model_name}_{field_name}")
        else:
            python_type = json_type_to_python(field_type)

        # Determine if field is optional
        is_required = field_name in required_fields
        if is_required:
            fields[field_name] = (python_type, Field(description=description))
        else:
            fields[field_name] = (Optional[python_type], Field(default=None, description=description))

    return create_model(model_name, **fields)


def build_json_schema(
    selected_fields: List[Dict[str, Any]],
    template_name: str,
    document_type: str,
    organization_id: str
) -> Dict[str, Any]:
    """
    Build a JSON schema from selected fields.

    Args:
        selected_fields: List of field definitions
        template_name: Name for the schema
        document_type: Type of document (invoice, contract, etc.)
        organization_id: Organization ID

    Returns:
        Complete JSON schema dict
    """
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": template_name,
        "type": "object",
        "properties": {},
        "required": [],
        "metadata": {
            "document_type": document_type,
            "created_at": datetime.utcnow().isoformat(),
            "organization_id": organization_id,
            "version": "1.0"
        }
    }

    line_item_fields = []
    header_fields = []

    # Separate line item fields from header fields
    for field in selected_fields:
        if field.get("location") == "line_item":
            line_item_fields.append(field)
        else:
            header_fields.append(field)

    # Add header/footer fields
    for field in header_fields:
        prop = {
            "type": _map_data_type_to_json(field.get("data_type", "string")),
            "description": field.get("display_name", field["field_name"])
        }
        schema["properties"][field["field_name"]] = prop
        if field.get("required", False):
            schema["required"].append(field["field_name"])

    # Add line items as array if present
    if line_item_fields:
        line_items_schema = {
            "type": "array",
            "description": "Line items from the document",
            "items": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
        for field in line_item_fields:
            prop = {
                "type": _map_data_type_to_json(field.get("data_type", "string")),
                "description": field.get("display_name", field["field_name"])
            }
            line_items_schema["items"]["properties"][field["field_name"]] = prop
            if field.get("required", False):
                line_items_schema["items"]["required"].append(field["field_name"])

        schema["properties"]["line_items"] = line_items_schema

    return schema


def _map_data_type_to_json(data_type: str) -> str:
    """Map field data types to JSON schema types."""
    mapping = {
        "string": "string",
        "text": "string",
        "number": "number",
        "integer": "integer",
        "float": "number",
        "currency": "number",
        "date": "string",  # ISO date string
        "datetime": "string",
        "boolean": "boolean",
        "bool": "boolean",
        "array": "array",
        "list": "array",
        "object": "object",
    }
    return mapping.get(data_type.lower(), "string")


# =============================================================================
# Content Utilities
# =============================================================================

def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of content for cache validation."""
    return hashlib.sha256(content.encode()).hexdigest()


def format_extraction_result(
    extracted_data: Dict[str, Any],
    document_name: str,
    schema_title: str,
    organization_id: str
) -> Dict[str, Any]:
    """Format extraction result with metadata."""
    return {
        "metadata": {
            "document_name": document_name,
            "schema_title": schema_title,
            "organization_id": organization_id,
            "extracted_at": datetime.utcnow().isoformat(),
        },
        "data": extracted_data
    }


def truncate_content(content: str, max_chars: int = 8000) -> str:
    """Truncate content to fit within LLM context window."""
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n[Content truncated...]"


# =============================================================================
# Tool Input Schemas
# =============================================================================

class FieldAnalyzerInput(BaseModel):
    """Input for field analyzer tool."""
    content: str = Field(description="Parsed document content to analyze")
    document_name: str = Field(description="Name of the source document")
    document_type_hint: Optional[str] = Field(
        default=None,
        description="Hint about document type (invoice, contract, receipt, purchase_order, etc.)"
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenant isolation"
    )


class SchemaGeneratorInput(BaseModel):
    """Input for schema generator tool."""
    selected_fields: List[Dict[str, Any]] = Field(
        description="Fields selected by user for extraction"
    )
    template_name: str = Field(
        description="Name for the schema template",
        min_length=1,
        max_length=100
    )
    document_type: str = Field(
        description="Type of document (invoice, contract, receipt, etc.)"
    )
    organization_id: str = Field(
        description="Organization ID for scoping"
    )
    save_to_gcs: bool = Field(
        default=True,
        description="Whether to save the schema to GCS"
    )


class DataExtractorInput(BaseModel):
    """Input for data extractor tool."""
    content: str = Field(description="Parsed document content")
    schema_definition: Dict[str, Any] = Field(description="JSON schema for extraction")
    document_name: str = Field(description="Source document name")
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization ID for multi-tenancy"
    )


# =============================================================================
# Output Schemas (for structured LLM output)
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


class FieldAnalysisResult(BaseModel):
    """Result of document field analysis."""
    document_type: str = Field(description="Detected document type (invoice, contract, receipt, etc.)")
    fields: List[DiscoveredField] = Field(description="List of discovered fields in header/footer")
    has_line_items: bool = Field(description="Whether document contains line items/rows")
    line_item_fields: Optional[List[DiscoveredField]] = Field(
        default=None,
        description="Fields within each line item (if has_line_items is True)"
    )


# =============================================================================
# Parallel Analysis Output Schemas
# =============================================================================

class HeaderFieldsResult(BaseModel):
    """Result of header/footer field analysis (parallel task 1)."""
    document_type: str = Field(description="Detected document type (invoice, contract, receipt, etc.)")
    fields: List[DiscoveredField] = Field(description="List of discovered header/footer fields")
    has_line_items: bool = Field(description="Whether document appears to contain line items/rows")


class LineItemFieldsResult(BaseModel):
    """Result of line item field analysis (parallel task 2)."""
    fields: List[DiscoveredField] = Field(
        default_factory=list,
        description="Fields within each line item"
    )
