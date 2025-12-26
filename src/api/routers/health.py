"""Health check API endpoint."""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter

from ..schemas.common import HealthStatus

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthStatus,
    operation_id="getHealth",
    summary="Check service health",
)
async def health_check():
    """
    Check the health of all service components.

    **No authentication required**: This endpoint does not require X-Organization-ID header.

    Returns status of:
    - Database connection
    - Document Agent (Google Gemini)
    - Sheets Agent (OpenAI GPT + DuckDB)
    - LlamaParse (document parsing)
    """
    components: Dict[str, Dict[str, Any]] = {}

    # Check database
    try:
        from src.db.connection import db
        engine = await db.get_engine_async()
        if engine:
            async with db.session() as session:
                if session:
                    await session.execute("SELECT 1")
                    components["database"] = {
                        "status": "healthy",
                        "message": "Connected"
                    }
                else:
                    components["database"] = {
                        "status": "disabled",
                        "message": "Database disabled"
                    }
        else:
            components["database"] = {
                "status": "unhealthy",
                "message": "No engine"
            }
    except Exception as e:
        components["database"] = {
            "status": "unhealthy",
            "message": str(e)
        }

    # Check Document Agent (use singleton from dependencies)
    try:
        from ..dependencies import get_document_agent
        agent = await get_document_agent()
        components["document_agent"] = {
            "status": "healthy",
            "message": "Available"
        }
    except Exception as e:
        components["document_agent"] = {
            "status": "unhealthy",
            "message": str(e)
        }

    # Check Sheets Agent (use singleton from dependencies)
    try:
        from ..dependencies import get_sheets_agent
        agent = await get_sheets_agent()
        health = agent.get_health_status()
        components["sheets_agent"] = {
            "status": health.get("status", "unknown"),
            "duckdb_ready": health.get("duckdb_ready", False),
        }
    except Exception as e:
        components["sheets_agent"] = {
            "status": "unhealthy",
            "message": str(e)
        }

    # Check LlamaParse
    try:
        from src.rag.llama_parse_util import parse_document
        components["llama_parse"] = {
            "status": "available",
            "message": "Module loaded"
        }
    except ImportError:
        components["llama_parse"] = {
            "status": "unavailable",
            "message": "Module not installed"
        }
    except Exception as e:
        components["llama_parse"] = {
            "status": "unhealthy",
            "message": str(e)
        }

    # Determine overall status
    unhealthy = any(
        c.get("status") == "unhealthy"
        for c in components.values()
    )

    return HealthStatus(
        status="unhealthy" if unhealthy else "healthy",
        version="3.0.0",
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get(
    "/",
    response_model=Dict[str, str],
    operation_id="getRoot",
    summary="Get API information",
)
async def root():
    """
    Root endpoint with API information and documentation links.

    **No authentication required**: This endpoint does not require X-Organization-ID header.
    """
    return {
        "service": "Document Intelligence AI",
        "version": "3.0.0",
        "docs": "/docs",
        "health": "/health",
    }
