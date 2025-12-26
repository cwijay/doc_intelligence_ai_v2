"""Custom middleware for logging, error handling, and request processing."""

import logging
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all incoming requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Log request
        start_time = time.time()
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- Client: {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Log response
        duration_ms = elapsed_ms(start_time)
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Duration: {duration_ms:.1f}ms"
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-MS"] = f"{duration_ms:.1f}"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling for unhandled exceptions."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "unknown")
            logger.exception(f"[{request_id}] Unhandled exception: {e}")

            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": "Internal server error",
                    "message": str(e) if logger.isEnabledFor(logging.DEBUG) else "An unexpected error occurred",
                    "request_id": request_id,
                }
            )


def add_middleware(app: FastAPI) -> None:
    """Add all custom middleware to the application."""
    # Order matters: Error handling should wrap request logging
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestLoggingMiddleware)


# =============================================================================
# Exception Handlers
# =============================================================================

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError


async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Debug: Log HTTP exception details
    logger.error(f"[{request_id}] HTTP {exc.status_code} - {exc.detail} - Path: {request.url.path}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id,
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed messages."""
    request_id = getattr(request.state, "request_id", "unknown")

    # Debug: Log the raw request body and validation errors
    try:
        body = await request.body()
        logger.error(f"[{request_id}] Validation error - Raw body: {body.decode('utf-8', errors='replace')}")
    except Exception as e:
        logger.error(f"[{request_id}] Validation error - Could not read body: {e}")

    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
        })
        logger.error(f"[{request_id}] Validation error - Field: {field}, Message: {error['msg']}")

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Validation error",
            "details": errors,
            "request_id": request_id,
        }
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers."""
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
