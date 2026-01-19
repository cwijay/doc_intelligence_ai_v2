"""
Unified Document Parser Interface

Provides a single interface for document parsing that can switch between
different parsing backends:
- LlamaParse: Full-featured parser with broad format support
- Gemini: Fast native Google AI parser with excellent OCR

Configuration via DOCUMENT_PARSER environment variable:
- "llamaparse" (default): Use LlamaParse
- "gemini": Use Gemini parser
"""

import logging
import os
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ParserBackend(str, Enum):
    """Available parser backends."""
    LLAMAPARSE = "llamaparse"
    GEMINI = "gemini"


# Configuration
DEFAULT_PARSER = os.getenv("DOCUMENT_PARSER", "gemini").lower()


def get_parser_backend() -> ParserBackend:
    """Get the configured parser backend."""
    backend = DEFAULT_PARSER
    if backend == "gemini":
        return ParserBackend.GEMINI
    return ParserBackend.LLAMAPARSE


async def parse_document_async(
    file_path: str,
    output_dir: Optional[str] = None,
    use_handwriting_mode: bool = False,
    complexity: str = "normal",
    parser: Optional[str] = None,
) -> str:
    """
    Parse a document using the configured parser backend.

    Args:
        file_path: Path to the document to parse
        output_dir: Optional directory to save the markdown file
        use_handwriting_mode: Use enhanced prompts for handwritten documents
        complexity: Document complexity ("normal" or "high")
        parser: Override parser backend ("llamaparse" or "gemini")

    Returns:
        str: Parsed document content in markdown format

    Example:
        >>> content = await parse_document_async("invoice.pdf")
        >>> content = await parse_document_async("doc.pdf", parser="gemini")
    """
    # Determine which parser to use
    backend = ParserBackend(parser) if parser else get_parser_backend()

    if backend == ParserBackend.GEMINI:
        from src.rag.gemini_parse_util import parse_document_async as gemini_parse
        return await gemini_parse(
            file_path=file_path,
            output_dir=output_dir,
            use_handwriting_mode=use_handwriting_mode,
            complexity=complexity,
        )
    else:
        from src.rag.llama_parse_util import parse_document_async as llama_parse
        return await llama_parse(
            file_path=file_path,
            output_dir=output_dir,
            use_agent_mode=use_handwriting_mode,
            complexity=complexity,
        )


def parse_document(
    file_path: str,
    output_dir: Optional[str] = None,
    use_handwriting_mode: bool = False,
    complexity: str = "normal",
    parser: Optional[str] = None,
) -> str:
    """
    Sync: Parse a document using the configured parser backend.

    Args:
        file_path: Path to the document to parse
        output_dir: Optional directory to save the markdown file
        use_handwriting_mode: Use enhanced prompts for handwritten documents
        complexity: Document complexity ("normal" or "high")
        parser: Override parser backend ("llamaparse" or "gemini")

    Returns:
        str: Parsed document content in markdown format
    """
    # Determine which parser to use
    backend = ParserBackend(parser) if parser else get_parser_backend()

    if backend == ParserBackend.GEMINI:
        from src.rag.gemini_parse_util import parse_document as gemini_parse
        return gemini_parse(
            file_path=file_path,
            output_dir=output_dir,
            use_handwriting_mode=use_handwriting_mode,
            complexity=complexity,
        )
    else:
        from src.rag.llama_parse_util import parse_document as llama_parse
        return llama_parse(
            file_path=file_path,
            output_dir=output_dir,
            use_agent_mode=use_handwriting_mode,
            complexity=complexity,
        )


def get_supported_extensions(parser: Optional[str] = None) -> list[str]:
    """
    Get supported file extensions for the specified parser.

    Args:
        parser: Parser backend ("llamaparse" or "gemini"), or None for configured default

    Returns:
        list[str]: List of supported file extensions
    """
    backend = ParserBackend(parser) if parser else get_parser_backend()

    if backend == ParserBackend.GEMINI:
        from src.rag.gemini_parse_util import SUPPORTED_EXTENSIONS
        return sorted(SUPPORTED_EXTENSIONS)
    else:
        from src.rag.llama_parse_util import SUPPORTED_EXTENSIONS
        return sorted(SUPPORTED_EXTENSIONS)
