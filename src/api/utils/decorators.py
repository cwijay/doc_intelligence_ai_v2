"""API utility decorators.

Provides reusable decorators for common patterns across API endpoints.
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, Type, TypeVar

from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)

T = TypeVar('T')


def with_timing(
    response_class: Optional[Type[T]] = None,
    timing_field: str = "processing_time_ms",
    error_field: str = "error",
    success_field: str = "success",
):
    """
    Decorator that automatically tracks processing time for endpoints.

    Injects processing_time_ms into the response object on success or failure.
    Works with any Pydantic response model that has a processing_time_ms field.

    Args:
        response_class: Optional response class to use for error responses.
                       If not provided, the original exception is re-raised.
        timing_field: Name of the timing field in the response (default: processing_time_ms)
        error_field: Name of the error field in the response (default: error)
        success_field: Name of the success field in the response (default: success)

    Usage:
        @router.post("/analyze")
        @with_timing(response_class=AnalyzeResponse)
        async def analyze(request: AnalyzeRequest):
            # ... processing logic
            return AnalyzeResponse(success=True, data=result)
            # processing_time_ms is automatically injected

    Note:
        The decorated function should return a response object that has
        a processing_time_ms field (or the field specified in timing_field).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Inject timing into response
                processing_time = elapsed_ms(start_time)

                if hasattr(result, timing_field):
                    setattr(result, timing_field, processing_time)
                elif isinstance(result, dict) and timing_field in result:
                    result[timing_field] = processing_time

                return result

            except Exception as e:
                processing_time = elapsed_ms(start_time)

                if response_class is not None:
                    # Return error response with timing
                    logger.exception(f"{func.__name__} failed: {e}")
                    return response_class(**{
                        success_field: False,
                        error_field: str(e),
                        timing_field: processing_time,
                    })
                else:
                    # Re-raise with timing logged
                    logger.exception(f"{func.__name__} failed after {processing_time}ms: {e}")
                    raise

        return wrapper
    return decorator


def log_errors(
    message_template: str = "{func_name} failed",
    log_level: str = "exception",
    reraise: bool = True,
):
    """
    Decorator for consistent error logging across endpoints.

    Args:
        message_template: Log message template (can include {func_name}, {error})
        log_level: Logging level to use (exception, error, warning)
        reraise: Whether to re-raise the exception after logging

    Usage:
        @router.post("/process")
        @log_errors(message_template="Document processing failed: {error}")
        async def process_document(request: Request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                message = message_template.format(
                    func_name=func.__name__,
                    error=str(e),
                )

                log_func = getattr(logger, log_level, logger.exception)
                log_func(message)

                if reraise:
                    raise

        return wrapper
    return decorator


def with_error_response(
    response_class: Type[T],
    success_field: str = "success",
    error_field: str = "error",
    timing_field: str = "processing_time_ms",
):
    """
    Decorator that wraps exceptions in a standardized error response.

    Combines timing tracking with error handling for a complete solution.

    Args:
        response_class: Response class to use for both success and error responses
        success_field: Name of the success field
        error_field: Name of the error field
        timing_field: Name of the timing field

    Usage:
        @router.post("/extract")
        @with_error_response(ExtractResponse)
        async def extract_data(request: ExtractRequest):
            result = await do_extraction(request)
            return ExtractResponse(success=True, data=result)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)

                # Inject timing into successful response
                processing_time = elapsed_ms(start_time)
                if hasattr(result, timing_field):
                    setattr(result, timing_field, processing_time)

                return result

            except Exception as e:
                processing_time = elapsed_ms(start_time)
                logger.exception(f"{func.__name__} failed: {e}")

                return response_class(**{
                    success_field: False,
                    error_field: str(e),
                    timing_field: processing_time,
                })

        return wrapper
    return decorator
