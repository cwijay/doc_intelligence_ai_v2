"""
Document Agent - AI-powered document analysis and content generation.

This module provides a ReAct agent for processing documents and generating:
- Summaries
- FAQs (Frequently Asked Questions)
- Comprehension questions

Usage:
    from src.agents.document import DocumentAgent, DocumentAgentConfig

    config = DocumentAgentConfig()
    agent = DocumentAgent(config)

    # Natural language interaction
    response = await agent.process_request(DocumentRequest(
        document_name="Sample1.md",
        query="Generate a summary and 5 FAQs for this document"
    ))

    # Direct methods
    summary = await agent.generate_summary("Sample1.md", max_words=300)
    faqs = await agent.generate_faqs("Sample1.md", num_faqs=5)
    questions = await agent.generate_questions("Sample1.md", num_questions=10)

    # All at once
    content = await agent.generate_all("Sample1.md")
"""

from .config import DocumentAgentConfig
from .core import DocumentAgent
from .schemas import (
    DocumentRequest,
    DocumentResponse,
    GenerationOptions,
    GeneratedContent,
    FAQ,
    Question,
    TokenUsage,
    DocumentMetadata,
    SessionInfo,
    ErrorResponse
)
from .tools import (
    DocumentLoaderTool,
    SummaryGeneratorTool,
    FAQGeneratorTool,
    QuestionGeneratorTool,
    ContentPersistTool,
    RAGSearchTool,
    create_document_tools
)

__all__ = [
    # Core
    "DocumentAgent",
    "DocumentAgentConfig",
    # Schemas
    "DocumentRequest",
    "DocumentResponse",
    "GenerationOptions",
    "GeneratedContent",
    "FAQ",
    "Question",
    "TokenUsage",
    "DocumentMetadata",
    "SessionInfo",
    "ErrorResponse",
    # Tools
    "DocumentLoaderTool",
    "SummaryGeneratorTool",
    "FAQGeneratorTool",
    "QuestionGeneratorTool",
    "ContentPersistTool",
    "RAGSearchTool",
    "create_document_tools",
]
