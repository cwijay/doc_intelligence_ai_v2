"""API utility functions."""

from .formatting import (
    format_duration_ms,
    format_time_ago,
    format_file_size,
    get_status_color,
    get_activity_title,
    get_activity_icon,
)

from .decorators import (
    with_timing,
    with_error_response,
    log_errors,
)

from .responses import (
    build_success_response,
    build_error_response,
    map_token_usage,
    map_list_items,
    ResponseBuilder,
)

__all__ = [
    # Formatting utilities
    "format_duration_ms",
    "format_time_ago",
    "format_file_size",
    "get_status_color",
    "get_activity_title",
    "get_activity_icon",
    # Decorators
    "with_timing",
    "with_error_response",
    "log_errors",
    # Response builders
    "build_success_response",
    "build_error_response",
    "map_token_usage",
    "map_list_items",
    "ResponseBuilder",
]
