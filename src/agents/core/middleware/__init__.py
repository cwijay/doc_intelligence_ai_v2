"""
Middleware module for agents.

Provides custom middleware components that fill gaps in LangChain:
- Query Classification: Intent detection for tool routing
- Tool Selection: LLM-based pre-filtering of tools

Note: Standard middleware (retry, fallback, call limits, PII detection) is now
provided by LangChain 1.2.0 built-in middleware. Use:
    from langchain.agents.middleware import (
        ModelRetryMiddleware,
        ToolRetryMiddleware,
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
        PIIMiddleware,
    )
"""

# Tool Selection (fills LangChain gap)
from .tool_selector import LLMToolSelector

# Query Classification (fills LangChain gap)
from .query_classifier import QueryClassifier, QueryIntent

__all__ = [
    "LLMToolSelector",
    "QueryClassifier",
    "QueryIntent",
]
