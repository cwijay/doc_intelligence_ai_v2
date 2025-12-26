"""Document processing tools package.

This package contains LangChain tools for document processing:
- DocumentLoaderTool: Load documents from GCS or local storage
- SummaryGeneratorTool: Generate document summaries using LLM
- FAQGeneratorTool: Generate FAQs from document content
- QuestionGeneratorTool: Generate comprehension questions
- ContentPersistTool: Persist generated content to GCS and PostgreSQL
- RAGSearchTool: Semantic search with Gemini File Search
"""

from typing import List

from langchain_core.tools import BaseTool

from ..config import DocumentAgentConfig

# Import all tools
from .document_loader import DocumentLoaderTool
from .summary_generator import SummaryGeneratorTool
from .faq_generator import FAQGeneratorTool
from .question_generator import QuestionGeneratorTool
from .persist import ContentPersistTool
from .rag_search import RAGSearchTool

# Import base utilities and schemas for external use
from .base import (
    # Input schemas
    DocumentLoaderInput,
    SummaryGeneratorInput,
    FAQGeneratorInput,
    QuestionGeneratorInput,
    ContentPersistInput,
    RAGSearchInput,
    # Utility functions
    derive_org_and_folder,
    derive_document_base,
    build_content_path,
    format_summary_markdown,
    format_faqs_json,
    format_questions_json,
    extract_llm_text,
    compute_content_hash,
)

# Backward compatibility alias (used by gcs_cache.py)
_build_content_path = build_content_path

__all__ = [
    # Tools
    "DocumentLoaderTool",
    "SummaryGeneratorTool",
    "FAQGeneratorTool",
    "QuestionGeneratorTool",
    "ContentPersistTool",
    "RAGSearchTool",
    # Factory function
    "create_document_tools",
    # Input schemas
    "DocumentLoaderInput",
    "SummaryGeneratorInput",
    "FAQGeneratorInput",
    "QuestionGeneratorInput",
    "ContentPersistInput",
    "RAGSearchInput",
    # Utilities
    "derive_org_and_folder",
    "derive_document_base",
    "build_content_path",
    "_build_content_path",  # Backward compatibility alias
    "format_summary_markdown",
    "format_faqs_json",
    "format_questions_json",
    "extract_llm_text",
    "compute_content_hash",
]


def create_document_tools(config: DocumentAgentConfig) -> List[BaseTool]:
    """Create all document processing tools with shared config.

    Args:
        config: DocumentAgentConfig with LLM settings, storage paths, etc.

    Returns:
        List of configured LangChain tools for document processing.
    """
    return [
        DocumentLoaderTool(config=config),
        SummaryGeneratorTool(config=config),
        FAQGeneratorTool(config=config),
        QuestionGeneratorTool(config=config),
        ContentPersistTool(config=config),
        RAGSearchTool(config=config),
    ]
