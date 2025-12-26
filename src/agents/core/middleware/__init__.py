"""
Middleware module for agents.

Provides production-ready middleware components for:
- Tool Selection: LLM-based pre-filtering of tools
- Resilience: Retry and fallback for model/tool failures
- Limits: Call tracking and enforcement
- Safety: PII detection and handling

Usage:
    from src.agents.core.middleware import create_middleware_stack, MiddlewareConfig

    # Use defaults
    stack = create_middleware_stack()

    # Or customize
    config = MiddlewareConfig(
        tool_selector_enabled=True,
        fallback_enabled=True,
        pii_enabled=True
    )
    stack = create_middleware_stack(config)
"""

# Configuration
from .config import MiddlewareConfig

# Call Limits
from .limits import (
    CallLimitCallbackHandler,
    CallLimitExceeded,
    CallLimitTracker,
)

# Resilience
from .resilience import (
    ModelFallback,
    ModelRetry,
    ToolRetry,
)

# Safety
from .safety import (
    PIIBlockedError,
    PIIDetector,
    PIIMatch,
    PIIStrategy,
    PIIType,
)

# Stack
from .stack import (
    MiddlewareStack,
    create_middleware_stack,
)

# Tool Selection
from .tool_selector import LLMToolSelector

# Query Classification
from .query_classifier import QueryClassifier, QueryIntent

__all__ = [
    # Configuration
    "MiddlewareConfig",
    # Tool Selection
    "LLMToolSelector",
    # Query Classification
    "QueryClassifier",
    "QueryIntent",
    # Resilience
    "ModelRetry",
    "ModelFallback",
    "ToolRetry",
    # Limits
    "CallLimitTracker",
    "CallLimitCallbackHandler",
    "CallLimitExceeded",
    # Safety
    "PIIDetector",
    "PIIStrategy",
    "PIIType",
    "PIIMatch",
    "PIIBlockedError",
    # Stack
    "MiddlewareStack",
    "create_middleware_stack",
]
