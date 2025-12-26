"""
Database utility functions and decorators.

Provides:
- Retry decorator for transient database errors
- Helper functions for common patterns
"""

import asyncio
import logging
from functools import wraps
from typing import TypeVar, Callable, Any

from sqlalchemy.exc import (
    OperationalError,
    InterfaceError,
    DisconnectionError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Define which exceptions are retryable
RETRYABLE_EXCEPTIONS = (
    OperationalError,
    InterfaceError,
    DisconnectionError,
    asyncio.TimeoutError,
    ConnectionRefusedError,
    ConnectionResetError,
)


def create_db_retry(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 10,
):
    """
    Create a tenacity retry decorator for database operations.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)

    Returns:
        Configured retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Default retry decorator for database operations
db_retry = create_db_retry()


def with_db_retry(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to add retry logic to async database functions.

    Usage:
        @with_db_retry
        async def my_db_operation():
            async with db.session() as session:
                ...

    Note: This decorator wraps the entire function, so the retry
    will re-execute the function from the beginning on failure.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        @db_retry
        async def inner():
            return await func(*args, **kwargs)
        return await inner()

    return wrapper


def serialize_datetime(obj: Any) -> Any:
    """
    Serialize datetime objects for JSON compatibility.

    Args:
        obj: Object to serialize

    Returns:
        ISO format string if datetime, otherwise original object
    """
    from datetime import datetime

    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def model_to_dict(model: Any, exclude_none: bool = False) -> dict:
    """
    Convert SQLAlchemy model to dictionary.

    Args:
        model: SQLAlchemy model instance
        exclude_none: If True, exclude keys with None values

    Returns:
        Dictionary representation of the model
    """
    from sqlalchemy.inspection import inspect

    result = {}
    mapper = inspect(model.__class__)

    for column in mapper.columns:
        value = getattr(model, column.key)
        value = serialize_datetime(value)

        if exclude_none and value is None:
            continue

        result[column.key] = value

    return result
