"""Extractor tools package.

Provides tools for document field analysis, schema generation, and data extraction.
"""

from typing import List

from langchain_core.tools import BaseTool

from ..config import ExtractorAgentConfig
from .field_analyzer import FieldAnalyzerTool
from .schema_generator import (
    SchemaGeneratorTool,
    load_schema_from_gcs,
    list_schemas_from_gcs,
    invalidate_schema_cache
)
from .data_extractor import DataExtractorTool


def create_extractor_tools(config: ExtractorAgentConfig) -> List[BaseTool]:
    """Create all extractor tools with the given configuration.

    Args:
        config: ExtractorAgentConfig instance

    Returns:
        List of configured BaseTool instances
    """
    return [
        FieldAnalyzerTool(config=config),
        SchemaGeneratorTool(config=config),
        DataExtractorTool(config=config),
    ]


__all__ = [
    "create_extractor_tools",
    "FieldAnalyzerTool",
    "SchemaGeneratorTool",
    "DataExtractorTool",
    "load_schema_from_gcs",
    "list_schemas_from_gcs",
    "invalidate_schema_cache",
]
