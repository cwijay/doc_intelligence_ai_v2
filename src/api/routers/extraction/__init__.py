"""Extraction API router package.

This package provides endpoints for document data extraction:
- Field analysis
- Schema generation
- Data extraction
- Template management
- Record management
- Export functionality

Organized into submodules for maintainability:
- analyze.py: Field analysis endpoint
- schema.py: Schema generation and template endpoints
- extract.py: Data extraction endpoint
- records.py: Record CRUD endpoints
- export.py: Export functionality
- health.py: Health check endpoint
- cache.py: GCS cache utilities
- helpers.py: Shared helper functions
"""

from fastapi import APIRouter

from .analyze import router as analyze_router
from .schema import router as schema_router
from .extract import router as extract_router
from .records import router as records_router
from .export import router as export_router
from .health import router as health_router

# Create main router that includes all sub-routers
router = APIRouter()

# Include all sub-routers
router.include_router(analyze_router)
router.include_router(schema_router)
router.include_router(extract_router)
router.include_router(records_router)
router.include_router(export_router)
router.include_router(health_router)

# Re-export for backwards compatibility
__all__ = ["router"]
