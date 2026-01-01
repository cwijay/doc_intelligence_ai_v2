"""
Shared document parsing service.
Used by both single file and bulk upload workflows.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.storage import get_storage
from src.utils.gcs_utils import is_gcs_path, extract_gcs_path_parts
from src.rag.llama_parse_util import parse_document as llama_parse
from src.db.repositories.audit_repository import register_or_update_parsed_document

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing a document."""
    success: bool
    parsed_content: Optional[str] = None
    parsed_path: Optional[str] = None
    error: Optional[str] = None
    parse_time_ms: int = 0
    estimated_pages: int = 0
    cached: bool = False


async def parse_and_save(
    file_path: str,
    folder_name: str,
    org_id: str,
    org_name: Optional[str] = None,
    save_to_gcs: bool = True,
    check_cache: bool = True,
) -> ParseResult:
    """
    Parse a document and optionally save to GCS.

    Handles:
    - GCS file download to temp location
    - Parsing with LlamaParse
    - Saving parsed content to GCS
    - Database registration

    Args:
        file_path: Local path or GCS URI (gs://...)
        folder_name: Target folder name for organization
        org_id: Organization ID for database
        org_name: Organization name for GCS path (extracted from file_path if not provided)
        save_to_gcs: Whether to save parsed content to GCS
        check_cache: Whether to check for already-parsed content

    Returns:
        ParseResult with parsed content and metadata
    """
    start_time = time.time()

    storage = get_storage()
    filename = Path(file_path).stem + ".md"

    # Extract org_name from file path if not provided
    if org_name is None:
        if is_gcs_path(file_path):
            path_parts = extract_gcs_path_parts(file_path)
            org_name = path_parts[1] if len(path_parts) > 1 else org_id
        else:
            org_name = org_id

    # Check cache for already-parsed content
    if check_cache and is_gcs_path(file_path):
        try:
            path_parts = extract_gcs_path_parts(file_path)
            bucket_name = path_parts[0] if path_parts else None

            if bucket_name:
                cached_path = f"gs://{bucket_name}/{org_name}/parsed/{folder_name}/{filename}"
                if await storage.exists(cached_path):
                    existing_content = await storage.read(cached_path)
                    if existing_content:
                        logger.info(f"Cache hit for parsed document: {cached_path}")
                        return ParseResult(
                            success=True,
                            parsed_content=existing_content,
                            parsed_path=cached_path,
                            cached=True,
                        )
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

    # Download GCS file to temp if needed
    temp_file_path = None
    local_file_path = file_path

    try:
        if is_gcs_path(file_path):
            # Check file exists
            if not await storage.exists(file_path):
                return ParseResult(
                    success=False,
                    error=f"File not found in GCS: {file_path}",
                )

            # Download to temp
            file_content = await storage.download_bytes(file_path)
            if file_content is None:
                return ParseResult(
                    success=False,
                    error=f"Failed to download file from GCS: {file_path}",
                )

            ext = Path(file_path).suffix
            temp_fd, temp_file_path = tempfile.mkstemp(suffix=ext)
            try:
                os.write(temp_fd, file_content)
            finally:
                os.close(temp_fd)

            local_file_path = temp_file_path
            logger.info(f"Downloaded GCS file to temp: {temp_file_path}")
        else:
            # Local file - validate exists
            if not os.path.exists(file_path):
                return ParseResult(
                    success=False,
                    error=f"File not found: {file_path}",
                )

        # Parse document
        try:
            loop = asyncio.get_running_loop()
            parsed_content = await loop.run_in_executor(
                None,
                lambda: llama_parse(file_path=local_file_path)
            )
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")

        parse_time_ms = int((time.time() - start_time) * 1000)

        if not parsed_content:
            return ParseResult(
                success=False,
                error="Parsing returned empty content",
                parse_time_ms=parse_time_ms,
            )

        # Estimate pages
        estimated_pages = max(1, len(parsed_content) // 3000)

        # Save to GCS if requested
        parsed_path = None
        if save_to_gcs:
            try:
                org_parsed_dir = f"{org_name}/parsed/{folder_name}"
                parsed_path = await storage.save(
                    parsed_content,
                    filename,
                    directory=org_parsed_dir,
                    use_prefix=False,
                )
                logger.info(f"Saved parsed content to GCS: {parsed_path}")

                # Register in database
                try:
                    await register_or_update_parsed_document(
                        storage_path=file_path,
                        filename=Path(file_path).name,
                        organization_id=org_id,
                        parsed_path=parsed_path,
                        folder_id=folder_name,
                    )
                except Exception as db_err:
                    logger.warning(f"Failed to register document in database: {db_err}")

            except Exception as e:
                logger.error(f"Failed to save to GCS: {e}")

        return ParseResult(
            success=True,
            parsed_content=parsed_content,
            parsed_path=parsed_path,
            parse_time_ms=parse_time_ms,
            estimated_pages=estimated_pages,
        )

    except Exception as e:
        parse_time_ms = int((time.time() - start_time) * 1000)
        logger.exception(f"Parse failed: {e}")
        return ParseResult(
            success=False,
            error=str(e),
            parse_time_ms=parse_time_ms,
        )
