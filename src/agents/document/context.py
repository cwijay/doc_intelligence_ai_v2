"""
Context variables for document agent execution.

Provides thread-safe and async-safe context for passing request-level
data to tools without relying on LLM parameter extraction.
"""

from contextvars import ContextVar
from typing import Optional, TypedDict
from contextlib import contextmanager


class RAGFilterContext(TypedDict, total=False):
    """Typed dict for RAG filter context."""
    file_filter: Optional[str]
    folder_filter: Optional[str]
    organization_id: Optional[str]


# Context variable for RAG filters - accessible from any tool during execution
_rag_filter_context: ContextVar[RAGFilterContext] = ContextVar(
    'rag_filter_context',
    default={}
)


def get_rag_filter_context() -> RAGFilterContext:
    """Get current RAG filter context.

    Returns:
        Dict with file_filter, folder_filter, organization_id (may be empty)
    """
    return _rag_filter_context.get()


def set_rag_filter_context(
    file_filter: Optional[str] = None,
    folder_filter: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> None:
    """Set RAG filter context for current execution.

    Args:
        file_filter: File name filter for cache scoping
        folder_filter: Folder name filter for cache scoping
        organization_id: Organization ID for multi-tenant isolation
    """
    _rag_filter_context.set({
        'file_filter': file_filter,
        'folder_filter': folder_filter,
        'organization_id': organization_id,
    })


def clear_rag_filter_context() -> None:
    """Clear RAG filter context."""
    _rag_filter_context.set({})


@contextmanager
def rag_filter_context(
    file_filter: Optional[str] = None,
    folder_filter: Optional[str] = None,
    organization_id: Optional[str] = None,
):
    """Context manager for RAG filter context.

    Usage:
        with rag_filter_context(file_filter="doc.pdf"):
            # RAG tool will use file_filter="doc.pdf" for cache operations
            await agent.invoke(...)
    """
    token = _rag_filter_context.set({
        'file_filter': file_filter,
        'folder_filter': folder_filter,
        'organization_id': organization_id,
    })
    try:
        yield
    finally:
        _rag_filter_context.reset(token)
