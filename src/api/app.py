"""FastAPI application factory."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .middleware import add_middleware, register_exception_handlers
from .routers import (
    documents_router,
    sheets_router,
    ingest_router,
    rag_router,
    audit_router,
    sessions_router,
    health_router,
)

logger = logging.getLogger(__name__)

# =============================================================================
# OpenAPI Configuration
# =============================================================================

OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Service health checks and status endpoints",
    },
    {
        "name": "Documents",
        "description": "Document Agent: Generate summaries, FAQs, and comprehension questions from documents using Google Gemini",
    },
    {
        "name": "Sheets",
        "description": "Sheets Agent: Analyze Excel/CSV files using natural language queries with DuckDB and OpenAI GPT",
    },
    {
        "name": "Ingestion",
        "description": "Upload and parse documents (PDF, DOCX, images with OCR support via LlamaParse)",
    },
    {
        "name": "RAG",
        "description": "Semantic search via Gemini File Store with folder organization for document retrieval",
    },
    {
        "name": "Audit",
        "description": "Processing job history, document generations, and audit trail for analytics",
    },
    {
        "name": "Sessions",
        "description": "Session management for stateful interactions with 30-minute timeout",
    },
]

API_DESCRIPTION = """
AI-powered document analysis system with two main agents:

## Document Agent
Generate summaries, FAQs, and comprehension questions from documents using Google Gemini.

## Sheets Agent
Analyze Excel/CSV files using natural language queries with DuckDB and OpenAI GPT.

## Additional Capabilities
- **Document Ingestion**: Upload and parse documents (PDF, DOCX, images with OCR via LlamaParse)
- **Semantic Search**: Gemini File Store for retrieval-augmented generation (RAG)
- **Audit Trail**: Full processing history and analytics

---

## Authentication

### Required Headers
- `X-Organization-ID`: Organization identifier for multi-tenant isolation (required for all endpoints except /health)

### Optional Headers
- `X-API-Key`: API key for authentication (required if `API_KEY_REQUIRED=true` in environment)
- `X-User-ID`: User identifier for personalization and audit logging

---

## Rate Limits
- **10 requests per 60 seconds** per session
- **Session timeout**: 30 minutes of inactivity

---

## Response Format
All endpoints return a consistent response format:
- `success`: Boolean indicating operation success
- `error`: Error message (only present on failure)
- `processing_time_ms`: Request processing time in milliseconds
- `token_usage`: LLM token consumption (when applicable)
"""


# Background cleanup task reference
_cleanup_task: asyncio.Task = None


async def _periodic_cleanup():
    """Background task to clean up expired sessions and rate limiter entries."""
    from .dependencies import get_document_agent, get_sheets_agent

    cleanup_interval = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "300"))  # 5 minutes default

    while True:
        try:
            await asyncio.sleep(cleanup_interval)

            # Cleanup document agent sessions
            try:
                doc_agent = await get_document_agent()
                doc_agent.session_manager.cleanup_expired_sessions()
                doc_agent.rate_limiter.cleanup()
            except Exception as e:
                logger.warning(f"Document agent cleanup error: {e}")

            # Cleanup sheets agent sessions
            try:
                sheets_agent = await get_sheets_agent()
                sheets_agent.session_manager.cleanup_expired_sessions()
                sheets_agent.rate_limiter.cleanup()
            except Exception as e:
                logger.warning(f"Sheets agent cleanup error: {e}")

            logger.debug("Periodic cleanup completed")

        except asyncio.CancelledError:
            logger.info("Periodic cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in periodic cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    global _cleanup_task

    # Startup
    logger.info("Starting Document Intelligence AI API...")

    # Initialize database connection pool
    try:
        from src.db.connection import db
        await db.get_engine_async()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")

    # Initialize agents at startup (fail-fast if any agent fails)
    from .dependencies import initialize_agents
    await initialize_agents()

    # Start periodic cleanup task
    _cleanup_task = asyncio.create_task(_periodic_cleanup())
    logger.info("Started periodic cleanup task")

    yield

    # Shutdown - order matters: cleanup task, agents, then database
    logger.info("Shutting down Document Intelligence AI API...")

    # 0. Cancel periodic cleanup task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        logger.info("Periodic cleanup task stopped")

    # 1. Shutdown agents first (they may have pending audit logs to write)
    try:
        from .dependencies import shutdown_agents
        await shutdown_agents()
        logger.info("Agents shutdown complete")
    except Exception as e:
        logger.warning(f"Agent shutdown error: {e}")

    # 2. Close database connections after agents are done
    try:
        from src.db.connection import db
        await db.close_all()
        logger.info("Database connections closed")
    except Exception as e:
        logger.warning(f"Database cleanup error: {e}")

    logger.info("Shutdown complete")


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with security schemes and server configuration."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )

    # Add server configuration
    openapi_schema["servers"] = [
        {"url": "http://localhost:8001", "description": "Local development server"},
    ]

    # Add security schemes
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    openapi_schema["components"]["securitySchemes"] = {
        "OrganizationId": {
            "type": "apiKey",
            "in": "header",
            "name": "X-Organization-ID",
            "description": "Organization ID for multi-tenant isolation (required for all endpoints except /health)",
        },
        "ApiKey": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication (optional, required if API_KEY_REQUIRED=true)",
        },
        "UserId": {
            "type": "apiKey",
            "in": "header",
            "name": "X-User-ID",
            "description": "User ID for personalization and audit logging (optional)",
        },
    }

    # Add global security requirement
    openapi_schema["security"] = [
        {"OrganizationId": []},
    ]

    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "Document Intelligence AI Support",
        "url": "https://github.com/biz2bricks/doc-intelligence-ai",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Get configuration from environment
    api_prefix = os.getenv("API_PREFIX", "/api/v1")
    debug = os.getenv("DEBUG", "false").lower() == "true"

    app = FastAPI(
        title="Document Intelligence AI",
        description=API_DESCRIPTION,
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        debug=debug,
    )

    # Set custom OpenAPI schema generator
    app.openapi = lambda: custom_openapi(app)

    # Add CORS middleware
    cors_origins = os.getenv("CORS_ORIGINS", '["*"]')
    try:
        import json
        origins = json.loads(cors_origins)
    except:
        origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware (logging, error handling)
    add_middleware(app)

    # Register custom exception handlers
    register_exception_handlers(app)

    # Include routers
    app.include_router(
        health_router,
        tags=["Health"],
    )

    app.include_router(
        documents_router,
        prefix=f"{api_prefix}/documents",
        tags=["Documents"],
    )

    app.include_router(
        sheets_router,
        prefix=f"{api_prefix}/sheets",
        tags=["Sheets"],
    )

    app.include_router(
        ingest_router,
        prefix=f"{api_prefix}/ingest",
        tags=["Ingestion"],
    )

    app.include_router(
        rag_router,
        prefix=f"{api_prefix}/rag",
        tags=["RAG"],
    )

    app.include_router(
        audit_router,
        prefix=f"{api_prefix}/audit",
        tags=["Audit"],
    )

    app.include_router(
        sessions_router,
        prefix=f"{api_prefix}/sessions",
        tags=["Sessions"],
    )

    return app
