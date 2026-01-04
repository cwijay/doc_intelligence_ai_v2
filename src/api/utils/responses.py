"""Response builder utilities.

Provides helper functions for building consistent API responses.
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def build_success_response(
    response_class: Type[T],
    data: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[float] = None,
    **kwargs,
) -> T:
    """
    Build a successful response with standard fields.

    Args:
        response_class: The Pydantic response class
        data: Additional data to include in the response
        processing_time_ms: Processing time in milliseconds
        **kwargs: Additional fields to set on the response

    Returns:
        Instance of response_class with success=True
    """
    response_data = {
        "success": True,
        **(data or {}),
        **kwargs,
    }

    if processing_time_ms is not None:
        response_data["processing_time_ms"] = processing_time_ms

    return response_class(**response_data)


def build_error_response(
    response_class: Type[T],
    error: str,
    processing_time_ms: Optional[float] = None,
    **kwargs,
) -> T:
    """
    Build an error response with standard fields.

    Args:
        response_class: The Pydantic response class
        error: Error message
        processing_time_ms: Processing time in milliseconds
        **kwargs: Additional fields to set on the response

    Returns:
        Instance of response_class with success=False
    """
    response_data = {
        "success": False,
        "error": error,
        **kwargs,
    }

    if processing_time_ms is not None:
        response_data["processing_time_ms"] = processing_time_ms

    return response_class(**response_data)


def map_token_usage(
    token_usage: Optional[Any],
    token_usage_class: Type[T],
) -> Optional[T]:
    """
    Map agent token usage to API schema.

    Args:
        token_usage: Token usage from agent response
        token_usage_class: Target Pydantic class for token usage

    Returns:
        Mapped token usage or None
    """
    if token_usage is None:
        return None

    try:
        return token_usage_class(
            prompt_tokens=getattr(token_usage, 'prompt_tokens', 0),
            completion_tokens=getattr(token_usage, 'completion_tokens', 0),
            total_tokens=getattr(token_usage, 'total_tokens', 0),
            estimated_cost_usd=getattr(token_usage, 'estimated_cost_usd', None),
        )
    except Exception as e:
        logger.warning(f"Failed to map token usage: {e}")
        return None


def map_list_items(
    items: Optional[List[Any]],
    target_class: Type[T],
    field_mapping: Optional[Dict[str, str]] = None,
) -> List[T]:
    """
    Map a list of items to a target Pydantic class.

    Args:
        items: Source items to map
        target_class: Target Pydantic class
        field_mapping: Optional mapping of source->target field names

    Returns:
        List of mapped items
    """
    if not items:
        return []

    result = []
    for item in items:
        try:
            if isinstance(item, dict):
                data = item
            else:
                data = item.__dict__ if hasattr(item, '__dict__') else {}

            # Apply field mapping if provided
            if field_mapping:
                mapped_data = {}
                for target_field, source_field in field_mapping.items():
                    if source_field in data:
                        mapped_data[target_field] = data[source_field]
                    elif hasattr(item, source_field):
                        mapped_data[target_field] = getattr(item, source_field)
                data = mapped_data

            result.append(target_class(**data))
        except Exception as e:
            logger.warning(f"Failed to map item to {target_class.__name__}: {e}")
            continue

    return result


class ResponseBuilder:
    """
    Fluent builder for constructing API responses.

    Usage:
        response = (
            ResponseBuilder(AnalyzeResponse)
            .success()
            .with_data(result=result_data)
            .with_timing(processing_time)
            .with_token_usage(agent_response.token_usage, TokenUsage)
            .build()
        )
    """

    def __init__(self, response_class: Type[T]):
        self.response_class = response_class
        self._data: Dict[str, Any] = {}

    def success(self, message: Optional[str] = None) -> "ResponseBuilder":
        """Mark response as successful."""
        self._data["success"] = True
        if message:
            self._data["message"] = message
        return self

    def error(self, error: str) -> "ResponseBuilder":
        """Mark response as failed with error message."""
        self._data["success"] = False
        self._data["error"] = error
        return self

    def with_data(self, **kwargs) -> "ResponseBuilder":
        """Add data fields to response."""
        self._data.update(kwargs)
        return self

    def with_timing(self, processing_time_ms: float) -> "ResponseBuilder":
        """Add processing time to response."""
        self._data["processing_time_ms"] = processing_time_ms
        return self

    def with_token_usage(
        self,
        token_usage: Optional[Any],
        token_usage_class: Type,
    ) -> "ResponseBuilder":
        """Add mapped token usage to response."""
        if token_usage:
            self._data["token_usage"] = map_token_usage(token_usage, token_usage_class)
        return self

    def build(self) -> T:
        """Build and return the response."""
        return self.response_class(**self._data)
