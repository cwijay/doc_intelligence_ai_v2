"""
RAG (Retrieval-Augmented Generation) package.

Combines document parsing, file search, and retrieval operations.
"""

# Gemini file store operations
from src.rag.gemini_file_store import (
    STORE_NAME,
    SOURCE_DIRECTORY,
    create_file_search_store,
    find_store_by_name,
    get_or_create_store,
    upload_file,
    upload_all_files,
    query_store,
    extract_citations,
    list_documents,
    delete_store,
    list_all_stores,
)

# Unified parser interface (recommended)
from src.rag.parser import (
    parse_document,
    parse_document_async,
    get_supported_extensions,
    ParserBackend,
    get_parser_backend,
)

# LlamaParse document parsing (direct access)
from src.rag.llama_parse_util import (
    parse_document as llama_parse_document,
    parse_document_async as llama_parse_document_async,
    parse_documents as llama_parse_documents,
    SUPPORTED_EXTENSIONS as LLAMA_SUPPORTED_EXTENSIONS,
)

# Gemini document parsing (direct access)
from src.rag.gemini_parse_util import (
    parse_document as gemini_parse_document,
    parse_document_async as gemini_parse_document_async,
    SUPPORTED_EXTENSIONS as GEMINI_SUPPORTED_EXTENSIONS,
)

# High-level service functions
from src.rag.file_search_service import (
    setup_store,
    search_by_document,
    search_all_documents,
    parse_handwritten_document,
)

__all__ = [
    # Gemini file store
    "STORE_NAME",
    "SOURCE_DIRECTORY",
    "create_file_search_store",
    "find_store_by_name",
    "get_or_create_store",
    "upload_file",
    "upload_all_files",
    "query_store",
    "extract_citations",
    "list_documents",
    "delete_store",
    "list_all_stores",
    # Unified parser (recommended)
    "parse_document",
    "parse_document_async",
    "get_supported_extensions",
    "ParserBackend",
    "get_parser_backend",
    # LlamaParse (direct access)
    "llama_parse_document",
    "llama_parse_document_async",
    "llama_parse_documents",
    "LLAMA_SUPPORTED_EXTENSIONS",
    # Gemini parser (direct access)
    "gemini_parse_document",
    "gemini_parse_document_async",
    "GEMINI_SUPPORTED_EXTENSIONS",
    # Service
    "setup_store",
    "search_by_document",
    "search_all_documents",
    "parse_handwritten_document",
]
