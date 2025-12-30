"""Extractor Agent package.

Provides AI-powered document field analysis and structured data extraction.
"""

from .config import ExtractorAgentConfig
from .core import ExtractorAgent
from .schemas import (
    TokenUsage,
    DiscoveredField,
    FieldSelection,
    AnalyzeFieldsRequest,
    AnalyzeFieldsResponse,
    GenerateSchemaRequest,
    GenerateSchemaResponse,
    ExtractDataRequest,
    ExtractDataResponse,
    TemplateInfo,
    TemplateListResponse,
    TemplateResponse,
    ExtractionRequest,
    ExtractionResponse,
)

__all__ = [
    # Main classes
    "ExtractorAgent",
    "ExtractorAgentConfig",
    # Token usage
    "TokenUsage",
    # Field analysis
    "DiscoveredField",
    "FieldSelection",
    "AnalyzeFieldsRequest",
    "AnalyzeFieldsResponse",
    # Schema generation
    "GenerateSchemaRequest",
    "GenerateSchemaResponse",
    # Data extraction
    "ExtractDataRequest",
    "ExtractDataResponse",
    # Templates
    "TemplateInfo",
    "TemplateListResponse",
    "TemplateResponse",
    # Full extraction
    "ExtractionRequest",
    "ExtractionResponse",
]
