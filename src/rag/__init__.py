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

# LlamaParse document parsing
from src.rag.llama_parse_util import (
    parse_document,
    parse_documents,
    get_supported_extensions,
    SUPPORTED_EXTENSIONS,
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
    # LlamaParse
    "parse_document",
    "parse_documents",
    "get_supported_extensions",
    "SUPPORTED_EXTENSIONS",
    # Service
    "setup_store",
    "search_by_document",
    "search_all_documents",
    "parse_handwritten_document",
]
