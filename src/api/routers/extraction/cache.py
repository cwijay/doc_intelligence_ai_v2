"""Extraction cache utilities for GCS-based caching.

Provides functions for building cache paths and reading/writing cached extractions.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def build_extraction_cache_path(
    org_name: str,
    folder_name: str,
    document_name: str,
    template_name: str
) -> str:
    """Build GCS path for extraction cache.

    Path format: {org_name}/extracted/{folder_name}/{doc_base}_{template}.json

    Args:
        org_name: Organization name (e.g., "Acme corp")
        folder_name: Folder name (e.g., "invoices")
        document_name: Document filename (e.g., "IMG_4694.md")
        template_name: Template name (e.g., "Invoice Template")

    Returns:
        GCS path like "Acme corp/extracted/invoices/IMG_4694_invoice_template.json"
    """
    doc_base = Path(document_name).stem
    safe_template = template_name.strip().replace(' ', '_').lower()
    parts = [org_name, "extracted"]
    if folder_name:
        parts.append(folder_name)
    parts.append(f"{doc_base}_{safe_template}.json")
    return '/'.join(parts)


def derive_folder_from_path(parsed_file_path: str) -> str:
    """Extract folder name from parsed file path.

    Example: "Acme corp/parsed/invoices/Sample1.md" -> "invoices"

    Args:
        parsed_file_path: Full path to parsed document

    Returns:
        Folder name or empty string if not found
    """
    if not parsed_file_path or 'parsed' not in parsed_file_path:
        return ""

    parts = parsed_file_path.split('/')
    try:
        parsed_idx = parts.index('parsed')
        folder_parts = parts[parsed_idx + 1:-1]  # Between 'parsed' and filename
        return '/'.join(folder_parts) if folder_parts else ""
    except ValueError:
        return ""


async def check_extraction_cache(cache_path: str) -> Optional[Dict[str, Any]]:
    """Check if extraction exists in GCS cache.

    Args:
        cache_path: GCS path to cached extraction

    Returns:
        Cached extraction data if found, None otherwise
    """
    try:
        from src.storage.config import get_storage

        storage = get_storage()
        content = await storage.read(cache_path, use_prefix=False)

        if content:
            cached_data = json.loads(content)
            logger.info(f"Extraction cache HIT: {cache_path}")
            return cached_data

        return None

    except Exception as e:
        logger.debug(f"Extraction cache miss: {e}")
        return None


async def save_extraction_cache(cache_path: str, extraction_data: Dict[str, Any]) -> bool:
    """Save extraction result to GCS cache.

    Args:
        cache_path: GCS path for cache
        extraction_data: Extracted data to cache

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        from src.storage.config import get_storage

        storage = get_storage()
        content = json.dumps(extraction_data, indent=2, default=str, ensure_ascii=False)

        filename = Path(cache_path).name
        directory = str(Path(cache_path).parent)

        await storage.save(
            content=content,
            filename=filename,
            directory=directory,
            use_prefix=False
        )
        logger.info(f"Extraction cached: {cache_path}")
        return True

    except Exception as e:
        logger.warning(f"Failed to save extraction cache: {e}")
        return False
