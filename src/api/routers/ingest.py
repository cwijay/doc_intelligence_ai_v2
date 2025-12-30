"""Document Ingestion API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
Files are stored in org-specific directories.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query

from src.core.executors import get_executors
from ..dependencies import get_upload_directory, get_parsed_directory, get_max_upload_size, get_org_id
from ..schemas.errors import FILE_ERROR_RESPONSES
from src.utils.timer_utils import elapsed_ms
from src.utils.gcs_utils import is_gcs_path, extract_gcs_path_parts
from ..usage import (
    check_quota,
    track_resource,
    check_token_limit_before_processing,
    check_resource_limit_before_processing,
    log_resource_usage_async,
)
from ..schemas.ingest import (
    ParseRequest,
    ParseResponse,
    ListFilesResponse,
    UploadResponse,
    UploadedFile,
    FileInfo,
    SaveAndIndexRequest,
    SaveAndIndexResponse,
)
from src.db.repositories.audit_repository import (
    register_uploaded_document,
    update_document_status,
    register_or_update_parsed_document,
    get_document_by_path,
)
from src.storage import get_storage, get_storage_config
from src.rag.llama_parse_util import parse_document as llama_parse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses=FILE_ERROR_RESPONSES,
    operation_id="uploadFile",
    summary="Upload files for processing",
)
async def upload_files(
    files: List[UploadFile] = File(...),
    upload_dir: str = Depends(get_upload_directory),
    max_size: int = Depends(get_max_upload_size),
    org_id: str = Depends(get_org_id),
):
    """
    Upload one or more files for processing.

    **Supported formats**: PDF, DOCX, PPTX, TXT, MD, XLSX, CSV, images (PNG, JPG, etc.)

    **Max file size**: 50MB per file (configurable via MAX_UPLOAD_SIZE_MB environment variable)

    **Multi-tenancy**: Files are stored in organization-specific directories.

    **Usage Tracking**: Storage bytes are tracked against the monthly storage limit.
    """
    # Multi-tenancy: Scope upload directory by organization
    org_upload_dir = os.path.join(upload_dir, org_id)
    os.makedirs(org_upload_dir, exist_ok=True)

    uploaded = []
    failed = []
    total_uploaded_bytes = 0

    for file in files:
        try:
            # Check file size
            content = await file.read()
            if len(content) > max_size:
                failed.append({
                    "filename": file.filename,
                    "error": f"File exceeds maximum size of {max_size // (1024*1024)}MB"
                })
                continue

            # Generate safe filename
            safe_filename = file.filename.replace(" ", "_")
            file_path = os.path.join(org_upload_dir, safe_filename)

            # Handle duplicate filenames
            base, ext = os.path.splitext(safe_filename)
            counter = 1
            while os.path.exists(file_path):
                safe_filename = f"{base}_{counter}{ext}"
                file_path = os.path.join(org_upload_dir, safe_filename)
                counter += 1

            # Save file
            with open(file_path, "wb") as f:
                f.write(content)

            # Register document in database with status='uploaded'
            try:
                await register_uploaded_document(
                    file_path=file_path,
                    file_name=file.filename,
                    file_size=len(content),
                    organization_id=org_id,
                )
            except Exception as e:
                logger.warning(f"Failed to register document in database: {e}")
                # Continue without failing - file was saved successfully

            file_size = len(content)
            total_uploaded_bytes += file_size

            uploaded.append(UploadedFile(
                filename=safe_filename,
                original_filename=file.filename,
                size_bytes=file_size,
                path=file_path,
                content_type=file.content_type,
                uploaded_at=datetime.utcnow(),
            ))

            logger.info(f"Uploaded file: {safe_filename} ({file_size} bytes)")

        except Exception as e:
            logger.error(f"Failed to upload {file.filename}: {e}")
            failed.append({
                "filename": file.filename,
                "error": str(e)
            })

    # Log storage usage for successfully uploaded files (non-blocking)
    if total_uploaded_bytes > 0:
        log_resource_usage_async(
            org_id=org_id,
            resource_type="storage_bytes",
            amount=total_uploaded_bytes,
            extra_data={
                "files_uploaded": len(uploaded),
                "filenames": [f.filename for f in uploaded],
            },
        )

    return UploadResponse(
        success=len(uploaded) > 0,
        files=uploaded,
        failed=failed,
        message=f"Uploaded {len(uploaded)} file(s), {len(failed)} failed"
    )


@router.post(
    "/parse",
    response_model=ParseResponse,
    responses=FILE_ERROR_RESPONSES,
    operation_id="parseDocument",
    summary="Parse document with LlamaParse",
)
async def parse_document(
    request: ParseRequest,
    parsed_dir: str = Depends(get_parsed_directory),
    org_id: str = Depends(get_org_id),
):
    """
    Parse a document using LlamaParse to extract content as Markdown.

    **Capabilities**:
    - Converts PDF, DOCX, images, and other formats to Markdown
    - OCR support for scanned documents
    - Handwriting recognition

    **Storage**: Parsed output can be automatically saved to GCS for later use.

    **Multi-tenancy**: Parsed output is stored in organization-specific GCS paths.

    **Quota**: Consumes LlamaParse pages from your subscription.
    """
    start_time = time.time()

    # Check LlamaParse page quota before processing
    try:
        if check_quota:
            from ..usage import check_token_limit_before_processing
            from src.core.usage import get_quota_checker

            checker = get_quota_checker()
            quota_result = await checker.check_quota(
                org_id=org_id,
                usage_type="llamaparse_pages",
                estimated_usage=1,  # Estimate 1 page per parse request
            )
            if not quota_result.allowed:
                from fastapi import HTTPException, status
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail={
                        "error": "quota_exceeded",
                        "usage_type": "llamaparse_pages",
                        "current_usage": quota_result.current_usage,
                        "limit": quota_result.limit,
                        "message": f"LlamaParse page limit exceeded. Used: {quota_result.current_usage}/{quota_result.limit}",
                        "upgrade": {
                            "tier": quota_result.upgrade_tier,
                            "message": quota_result.upgrade_message,
                            "url": quota_result.upgrade_url,
                        } if quota_result.upgrade_tier else None,
                    }
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"LlamaParse quota check failed, proceeding anyway: {e}")

    # Debug: Log incoming request
    logger.info(f"Parse request received: file_path={request.file_path}, folder_name={request.folder_name}, org_id={org_id}")

    # Check if document is already parsed (cache check)
    try:
        storage = get_storage()
        filename = Path(request.file_path).stem + ".md"

        # Extract bucket name and org name from file path (no hardcoding)
        # e.g., gs://bucket/ACME corp/original/invoices/Sample1.pdf -> bucket="bucket", org_name="ACME corp"
        if is_gcs_path(request.file_path):
            path_parts = extract_gcs_path_parts(request.file_path)
            bucket_name = path_parts[0] if path_parts else None
            org_name = path_parts[1] if len(path_parts) > 1 else org_id
        else:
            # For local files, skip cache check
            bucket_name = None
            org_name = org_id

        # Only check cache for GCS files
        if bucket_name:
            # Expected parsed file path: {org_name}/parsed/{folder_name}/{filename}.md
            gcs_uri = f"gs://{bucket_name}/{org_name}/parsed/{request.folder_name}/{filename}"

            if await storage.exists(gcs_uri):
                # Read existing content
                existing_content = await storage.read(gcs_uri)
                if existing_content:
                    logger.info(f"Document already parsed, returning cached: {gcs_uri}")
                    return ParseResponse(
                        success=True,
                        file_path=request.file_path,
                        output_path=gcs_uri,
                        parsed_content=existing_content,
                        content_preview=existing_content[:500] if len(existing_content) > 500 else existing_content,
                        pages=None,
                        format=request.output_format,
                        extraction_time_ms=0,  # No parsing time since cached
                    )
    except Exception as e:
        logger.warning(f"Cache check failed, proceeding with parsing: {e}")

    try:
        local_file_path = None
        temp_file_path = None

        # Handle GCS URIs vs local files
        if is_gcs_path(request.file_path):
            # Download from GCS to temp file for parsing
            storage = get_storage()

            # Check if file exists in GCS
            if not await storage.exists(request.file_path):
                return ParseResponse(
                    success=False,
                    file_path=request.file_path,
                    format=request.output_format,
                    extraction_time_ms=0,
                    error=f"File not found in GCS: {request.file_path}"
                )

            # Download file content as bytes
            file_content = await storage.download_bytes(request.file_path)
            if file_content is None:
                return ParseResponse(
                    success=False,
                    file_path=request.file_path,
                    format=request.output_format,
                    extraction_time_ms=0,
                    error=f"Failed to download file from GCS: {request.file_path}"
                )

            # Create temp file with original extension
            ext = Path(request.file_path).suffix
            temp_fd, temp_file_path = tempfile.mkstemp(suffix=ext)
            try:
                os.write(temp_fd, file_content)
            finally:
                os.close(temp_fd)

            local_file_path = temp_file_path
            logger.info(f"Downloaded GCS file to temp: {temp_file_path} ({len(file_content)} bytes)")
        else:
            # Local file - validate it exists
            if not os.path.exists(request.file_path):
                return ParseResponse(
                    success=False,
                    file_path=request.file_path,
                    format=request.output_format,
                    extraction_time_ms=0,
                    error=f"File not found: {request.file_path}"
                )
            local_file_path = request.file_path

        # Parse document (using local file path)
        # llama_parse is a sync function that returns markdown string directly
        try:
            loop = asyncio.get_running_loop()
            parsed_content = await loop.run_in_executor(
                None,
                lambda: llama_parse(file_path=local_file_path)
            )
        finally:
            # Clean up temp file if we created one
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temp file: {temp_file_path}")

        extraction_time = elapsed_ms(start_time)

        if not parsed_content:
            return ParseResponse(
                success=False,
                file_path=request.file_path,
                format=request.output_format,
                extraction_time_ms=extraction_time,
                error="Failed to parse document - no content extracted"
            )
        output_path = None

        # Save to GCS storage if requested
        if request.save_to_parsed:
            try:
                storage = get_storage()
                filename = Path(request.file_path).stem + ".md"

                # Extract org name from the file path
                # e.g., gs://bucket/ACME corp/original/invoices/Sample1.pdf -> "ACME corp"
                if is_gcs_path(request.file_path):
                    path_parts = extract_gcs_path_parts(request.file_path, max_splits=2)
                    org_name = path_parts[1] if len(path_parts) > 1 else org_id
                else:
                    org_name = org_id

                # Path structure: {org_name}/parsed/{folder_name}/{filename}.md
                org_parsed_dir = f"{org_name}/parsed/{request.folder_name}"

                output_path = await storage.save(
                    parsed_content,
                    filename,
                    directory=org_parsed_dir,
                    use_prefix=False  # Save directly without GCS prefix
                )

                logger.info(f"Saved parsed content to GCS: {output_path}")

                # Register/update document with parsed status in database
                try:
                    await register_or_update_parsed_document(
                        storage_path=request.file_path,
                        filename=Path(request.file_path).name,
                        organization_id=org_id,
                        parsed_path=output_path,
                        folder_id=request.folder_name,
                    )
                except Exception as db_err:
                    logger.warning(f"Failed to register/update document status in database: {db_err}")
                    # Continue without failing - parsed content was saved successfully

            except Exception as e:
                logger.error(f"Failed to save to GCS: {e}")
                # Continue without failing the whole request
                output_path = None

        # Track LlamaParse page usage after successful parsing
        try:
            if track_resource:
                from src.core.usage import get_usage_service

                service = get_usage_service()
                # Estimate page count based on content length (roughly 3000 chars per page)
                estimated_pages = max(1, len(parsed_content) // 3000)
                asyncio.create_task(
                    service.log_resource_usage(
                        org_id=org_id,
                        resource_type="llamaparse_pages",
                        amount=estimated_pages,
                        file_name=Path(request.file_path).name,
                        metadata={"file_path": request.file_path, "folder_name": request.folder_name},
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to log LlamaParse usage: {e}")

        return ParseResponse(
            success=True,
            file_path=request.file_path,
            output_path=output_path,
            parsed_content=parsed_content,
            content_preview=parsed_content[:500] if len(parsed_content) > 500 else parsed_content,
            pages=None,  # LlamaParse returns content string only, not page count
            format=request.output_format,
            extraction_time_ms=extraction_time,
        )

    except ImportError:
        return ParseResponse(
            success=False,
            file_path=request.file_path,
            format=request.output_format,
            extraction_time_ms=elapsed_ms(start_time),
            error="LlamaParse module not available"
        )
    except Exception as e:
        logger.exception(f"Document parsing failed: {e}")
        return ParseResponse(
            success=False,
            file_path=request.file_path,
            format=request.output_format,
            extraction_time_ms=elapsed_ms(start_time),
            error=str(e)
        )


@router.get(
    "/files",
    response_model=ListFilesResponse,
    responses=FILE_ERROR_RESPONSES,
    operation_id="listFiles",
    summary="List files in directory",
)
async def list_files(
    directory: str = Query(default="parsed", description="Directory to list: 'parsed' or 'upload'"),
    extension: str = Query(default=None, description="Filter by extension (e.g., '.md')"),
    upload_dir: str = Depends(get_upload_directory),
    parsed_dir: str = Depends(get_parsed_directory),
    org_id: str = Depends(get_org_id),
):
    """
    List available files in the specified directory.

    **Directories**:
    - `parsed`: Pre-parsed documents in GCS (Markdown format)
    - `upload`: Raw uploaded files in local storage

    **Multi-tenancy**: Only lists files belonging to the requesting organization.
    """
    try:
        files = []

        if directory == "parsed":
            # List from GCS storage (scoped by organization)
            try:
                storage = get_storage()
                storage_config = get_storage_config()

                # Multi-tenancy: Path structure is {org_id}/parsed/
                org_parsed_dir = f"{org_id}/parsed"

                gcs_files = await storage.list_files(
                    org_parsed_dir,
                    extension=extension
                )

                for gcs_path in gcs_files:
                    name = gcs_path.split("/")[-1]
                    files.append(FileInfo(
                        name=name,
                        path=gcs_path,
                        size_bytes=0,  # Size not easily available without extra GCS call
                        extension=Path(name).suffix,
                        modified_at=datetime.utcnow(),  # Timestamp not available without extra call
                        is_parsed=True,
                        status="parsed",  # Files in parsed directory are parsed
                        parsed_path=gcs_path,
                    ))

            except Exception as e:
                logger.error(f"Failed to list GCS files: {e}")
                return ListFilesResponse(
                    success=False,
                    directory=directory,
                    error=f"Failed to list GCS files: {e}"
                )

        elif directory == "upload":
            # List from local upload directory (scoped by organization)
            # Multi-tenancy: Scope upload directory by organization
            org_upload_dir = os.path.join(upload_dir, org_id)

            if not os.path.exists(org_upload_dir):
                os.makedirs(org_upload_dir, exist_ok=True)

            # Move filesystem operations to executor to avoid blocking event loop
            loop = asyncio.get_running_loop()

            def _get_file_items():
                """Get file items with stats in executor thread."""
                items = []
                for item in Path(org_upload_dir).iterdir():
                    if item.is_file():
                        if extension and not item.suffix.lower() == extension.lower():
                            continue
                        items.append((item, item.stat()))
                return items

            file_items = await loop.run_in_executor(
                get_executors().io_executor,
                _get_file_items
            )

            # Parallelize DB lookups with asyncio.gather()
            async def _get_doc_status(file_path: str):
                """Get document status from database."""
                try:
                    doc_info = await get_document_by_path(file_path, org_id)
                    if doc_info:
                        return {
                            "status": doc_info.get("status", "uploaded"),
                            "parsed_path": doc_info.get("parsed_path"),
                            "parsed_at": doc_info.get("parsed_at"),
                        }
                except Exception as e:
                    logger.debug(f"Could not get document status for {file_path}: {e}")
                return {"status": "uploaded", "parsed_path": None, "parsed_at": None}

            # Run all DB lookups concurrently
            file_paths = [str(item) for item, _ in file_items]
            doc_statuses = await asyncio.gather(*[_get_doc_status(fp) for fp in file_paths])

            # Build file info list
            for (item, stat), doc_status in zip(file_items, doc_statuses):
                files.append(FileInfo(
                    name=item.name,
                    path=str(item),
                    size_bytes=stat.st_size,
                    extension=item.suffix,
                    modified_at=datetime.fromtimestamp(stat.st_mtime),
                    is_parsed=(doc_status["status"] == "parsed"),
                    status=doc_status["status"],
                    parsed_path=doc_status["parsed_path"],
                    parsed_at=doc_status["parsed_at"],
                ))

            # Sort by modified time (newest first)
            files.sort(key=lambda x: x.modified_at, reverse=True)

        else:
            return ListFilesResponse(
                success=False,
                directory=directory,
                error=f"Invalid directory: {directory}. Use 'parsed' or 'upload'"
            )

        return ListFilesResponse(
            success=True,
            directory=directory,
            files=files,
            count=len(files),
        )

    except Exception as e:
        logger.exception(f"Failed to list files: {e}")
        return ListFilesResponse(
            success=False,
            directory=directory,
            error=str(e)
        )


@router.post(
    "/save-and-index",
    response_model=SaveAndIndexResponse,
    responses=FILE_ERROR_RESPONSES,
    operation_id="saveAndIndex",
    summary="Save parsed content to GCS and index in Gemini File Search",
)
async def save_and_index(
    request: SaveAndIndexRequest,
    org_id: str = Depends(get_org_id),
):
    """
    Save parsed/edited content to GCS and index in Gemini File Search store.

    This is a facade endpoint that orchestrates two existing operations:
    1. Save content to GCS at the specified target path
    2. Upload to Gemini File Search store for semantic retrieval

    **Store naming convention**: <org_name>_file_search_store

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    start_time = time.time()
    saved_path = None
    store_info = None
    indexed = False

    try:
        # Step 1: Save content to GCS using existing storage module
        storage = get_storage()
        storage_config = get_storage_config()

        # Extract filename from target_path
        target_filename = os.path.basename(request.target_path)

        # Construct directory using org_name and folder_name (consistent with load-parsed endpoint)
        target_dir = f"{request.org_name}/parsed/{request.folder_name}"

        logger.info(f"Saving content to GCS: {target_dir}/{target_filename}")

        saved_path = await storage.save(
            content=request.content,
            filename=target_filename,
            directory=target_dir,
            use_prefix=False  # Don't add GCS prefix, use path as-is
        )

        logger.info(f"Content saved to GCS: {saved_path}")

        # Step 2: Get or create Gemini store using existing function from rag_helpers
        from src.api.routers.rag_helpers import get_or_create_org_store

        store_info = await get_or_create_org_store(org_id, request.org_name)
        logger.info(f"Using Gemini store: {store_info['id']} ({store_info['display_name']})")

        # Step 3: Upload to Gemini store using existing function
        from src.rag.gemini_file_store import upload_file
        from google import genai
        from datetime import datetime as dt

        # Get the Gemini store object (use executor to avoid blocking event loop)
        gemini_store_id = store_info.get("gemini_store_id")
        gemini_store = None
        if gemini_store_id:
            client = genai.Client()
            loop = asyncio.get_running_loop()
            stores = await loop.run_in_executor(
                get_executors().io_executor,
                lambda: list(client.file_search_stores.list())
            )
            gemini_store = next((s for s in stores if s.name == gemini_store_id), None)

        if gemini_store:
            # saved_path already contains full GCS URI from storage.save()
            full_gcs_path = saved_path

            # Extract original file extension
            original_ext = None
            if request.original_filename:
                original_ext = os.path.splitext(request.original_filename)[1]

            # Current timestamp for parse_date
            parse_date = dt.utcnow().isoformat()

            upload_file(
                gemini_store,
                full_gcs_path,
                organization_id=org_id,
                folder_name=request.folder_name,
                org_name=request.org_name,
                original_gcs_path=request.original_gcs_path,
                parsed_gcs_path=full_gcs_path,
                original_file_extension=original_ext,
                parse_date=parse_date,
                parser_version=request.parser_version,
            )

            indexed = True
            logger.info(f"Content indexed in Gemini store: {store_info['display_name']}")
        else:
            logger.warning(f"Could not find Gemini store object for: {gemini_store_id}")

        processing_time = elapsed_ms(start_time)

        return SaveAndIndexResponse(
            success=True,
            saved_path=f"gs://{storage_config.gcs_bucket}/{saved_path}" if saved_path else None,
            store_id=store_info["id"] if store_info else None,
            store_name=store_info["display_name"] if store_info else None,
            indexed=indexed,
            message=f"Document saved and {'indexed' if indexed else 'saved (indexing skipped)'} successfully in {processing_time:.0f}ms"
        )

    except Exception as e:
        logger.exception(f"Save and index failed: {e}")
        return SaveAndIndexResponse(
            success=False,
            saved_path=saved_path,
            store_id=store_info["id"] if store_info else None,
            store_name=store_info["display_name"] if store_info else None,
            indexed=indexed,
            error=str(e)
        )
