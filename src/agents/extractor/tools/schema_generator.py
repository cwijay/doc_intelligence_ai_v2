"""Schema generator tool for extraction templates.

Generates JSON schemas from selected fields.
Note: GCS save is handled by the async caller (ExtractorAgent.generate_schema).
"""

import json
import logging
import time
import threading
from functools import lru_cache
from typing import Optional, Type, List, Dict, Any, Tuple

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from ..config import ExtractorAgentConfig
from .base import (
    SchemaGeneratorInput,
    build_json_schema,
    build_schema_path
)
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)

# =============================================================================
# TEMPLATE SCHEMA CACHE
# =============================================================================

# Cache configuration
SCHEMA_CACHE_MAX_SIZE = 50  # Max cached schemas
SCHEMA_CACHE_TTL_SECONDS = 300  # 5 minutes

# Thread-safe cache with TTL
_schema_cache: Dict[str, Tuple[Dict[str, Any], float]] = {}
_schema_cache_lock = threading.Lock()


def _get_cache_key(organization_id: str, template_name: str) -> str:
    """Generate cache key for schema."""
    safe_name = template_name.strip().replace(' ', '_').lower()
    return f"{organization_id}:{safe_name}"


def _get_cached_schema(organization_id: str, template_name: str) -> Optional[Dict[str, Any]]:
    """Get schema from cache if valid (not expired)."""
    key = _get_cache_key(organization_id, template_name)
    with _schema_cache_lock:
        if key in _schema_cache:
            schema, cached_at = _schema_cache[key]
            if time.time() - cached_at < SCHEMA_CACHE_TTL_SECONDS:
                logger.debug(f"Schema cache HIT: {template_name}")
                return schema
            else:
                # Expired - remove from cache
                del _schema_cache[key]
                logger.debug(f"Schema cache EXPIRED: {template_name}")
    return None


def _set_cached_schema(organization_id: str, template_name: str, schema: Dict[str, Any]) -> None:
    """Store schema in cache with TTL."""
    key = _get_cache_key(organization_id, template_name)
    with _schema_cache_lock:
        # Evict oldest entries if cache is full
        if len(_schema_cache) >= SCHEMA_CACHE_MAX_SIZE:
            oldest_key = min(_schema_cache.keys(), key=lambda k: _schema_cache[k][1])
            del _schema_cache[oldest_key]
            logger.debug(f"Schema cache evicted: {oldest_key}")

        _schema_cache[key] = (schema, time.time())
        logger.debug(f"Schema cached: {template_name}")


def invalidate_schema_cache(organization_id: str, template_name: Optional[str] = None) -> None:
    """Invalidate cached schema(s).

    Args:
        organization_id: Organization ID
        template_name: Specific template to invalidate, or None to invalidate all for org
    """
    with _schema_cache_lock:
        if template_name:
            key = _get_cache_key(organization_id, template_name)
            if key in _schema_cache:
                del _schema_cache[key]
                logger.debug(f"Schema cache invalidated: {template_name}")
        else:
            # Invalidate all schemas for this org
            prefix = f"{organization_id}:"
            keys_to_remove = [k for k in _schema_cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del _schema_cache[key]
            if keys_to_remove:
                logger.debug(f"Schema cache invalidated: {len(keys_to_remove)} schemas for org {organization_id}")


class SchemaGeneratorTool(BaseTool):
    """Tool to generate and save extraction schema templates.

    Takes user-selected fields and creates a JSON schema that can be
    used for structured data extraction. Optionally saves to GCS.
    """

    name: str = "schema_generator"
    description: str = """Generate a JSON schema from selected fields and optionally save as a template.
    Takes selected fields, template name, and document type.
    Returns the generated schema and GCS URI if saved."""
    args_schema: Type[BaseModel] = SchemaGeneratorInput

    config: ExtractorAgentConfig = Field(default_factory=ExtractorAgentConfig)

    def _run(
        self,
        selected_fields: List[Dict[str, Any]],
        template_name: str,
        document_type: str,
        organization_id: str,
        save_to_gcs: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate JSON schema from selected fields."""
        start_time = time.time()

        try:
            # Validate inputs
            if not selected_fields:
                return json.dumps({
                    "success": False,
                    "error": "No fields selected for schema generation"
                })

            if not template_name or not template_name.strip():
                return json.dumps({
                    "success": False,
                    "error": "Template name is required"
                })

            # Build the JSON schema
            schema = build_json_schema(
                selected_fields=selected_fields,
                template_name=template_name.strip(),
                document_type=document_type,
                organization_id=organization_id
            )

            duration_ms = elapsed_ms(start_time)

            # Count fields in schema
            header_fields = len([f for f in selected_fields if f.get("location") != "line_item"])
            line_item_fields = len([f for f in selected_fields if f.get("location") == "line_item"])

            logger.info(
                f"Schema generated: {template_name} "
                f"({header_fields} header, {line_item_fields} line item fields) "
                f"in {duration_ms:.1f}ms"
            )

            return json.dumps({
                "success": True,
                "template_name": template_name.strip(),
                "document_type": document_type,
                "schema": schema,
                "save_to_gcs": save_to_gcs,  # Flag for caller to handle GCS save
                "organization_id": organization_id,
                "header_fields_count": header_fields,
                "line_item_fields_count": line_item_fields,
                "processing_time_ms": duration_ms
            }, default=str)

        except Exception as e:
            logger.error(f"Schema generation failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e),
                "processing_time_ms": elapsed_ms(start_time)
            })


async def load_schema_from_gcs(
    organization_id: str,
    template_name: str,
    folder_name: Optional[str] = None,
    use_cache: bool = True
) -> Optional[Dict[str, Any]]:
    """Load a schema template from GCS with caching.

    Uses org_name and folder_name to match the save path structure:
    {org_name}/schema/{folder_name}/{template_name}.json

    Args:
        organization_id: Organization ID (UUID)
        template_name: Template name
        folder_name: Optional folder name where schema is stored
        use_cache: Whether to use the in-memory cache (default: True)

    Returns:
        Schema dict if found, None otherwise
    """
    # Check cache first
    if use_cache:
        cached = _get_cached_schema(organization_id, template_name)
        if cached is not None:
            logger.info(f"Schema loaded from cache: {template_name}")
            return cached

    try:
        from src.storage.config import get_storage
        from src.db.repositories.extraction_repository import get_organization_name

        storage = get_storage()
        safe_name = template_name.strip().replace(' ', '_').lower()

        # Resolve org_id to org_name for path construction
        org_name = await get_organization_name(organization_id)
        if not org_name:
            logger.warning(f"Could not resolve org_name for {organization_id}, using org_id as fallback")
            org_name = organization_id

        # Build path matching save structure: {org_name}/schema/{folder_name}/{template}.json
        if folder_name:
            path = f"{org_name}/schema/{folder_name}/{safe_name}.json"
        else:
            path = f"{org_name}/schema/{safe_name}.json"

        logger.info(f"Loading schema from GCS: {path}")
        content = await storage.read(path, use_prefix=False)

        if content:
            schema = json.loads(content)
            # Cache the schema
            if use_cache:
                _set_cached_schema(organization_id, template_name, schema)
            logger.info(f"Schema loaded from GCS: {template_name}")
            return schema

        logger.warning(f"Schema not found at path: {path}")
        return None

    except Exception as e:
        logger.error(f"Failed to load schema from GCS: {e}")
        return None


async def list_schemas_from_gcs(organization_id: str) -> List[Dict[str, Any]]:
    """List all schema templates for an organization from GCS.

    Uses org_name to match the save path structure:
    {org_name}/schema/**/*.json

    Args:
        organization_id: Organization ID (UUID)

    Returns:
        List of schema metadata dicts
    """
    try:
        from src.storage.config import get_storage
        from src.db.repositories.extraction_repository import get_organization_name

        storage = get_storage()

        # Resolve org_id to org_name for path construction
        org_name = await get_organization_name(organization_id)
        if not org_name:
            logger.warning(f"Could not resolve org_name for {organization_id}, using org_id as fallback")
            org_name = organization_id

        # List from org_name/schema directory
        directory = f"{org_name}/schema"
        logger.info(f"Listing schemas from GCS: {directory}")
        files = await storage.list_files(directory, extension=".json", use_prefix=False)

        schemas = []
        for file_path in files:
            try:
                content = await storage.read(file_path, use_prefix=False)
                if content:
                    schema = json.loads(content)
                    schemas.append({
                        "name": schema.get("title", file_path),
                        "document_type": schema.get("metadata", {}).get("document_type", "unknown"),
                        "created_at": schema.get("metadata", {}).get("created_at"),
                        "gcs_path": file_path,
                        "field_count": len(schema.get("properties", {}))
                    })
            except Exception as e:
                logger.warning(f"Failed to read schema {file_path}: {e}")

        logger.info(f"Found {len(schemas)} schemas for org {org_name}")
        return schemas

    except Exception as e:
        logger.error(f"Failed to list schemas from GCS: {e}")
        return []
