"""
Gemini File Search Store - Full CRUD Operations

This module provides functionality to:
- Create a file search store (one per organization)
- Upload files with chunking and metadata (including folder info)
- Query the store with optional folder/file filtering
- Extract citations from responses
- List documents in the store
- Delete the store

Multi-tenancy Architecture:
- One store per organization
- Documents tagged with folder metadata for filtering
- Supports cross-folder and folder-specific searches
"""

import os
import time
import glob
import datetime
import logging
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Dict, Tuple, Any, Optional, List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.gcs_utils import is_gcs_path, parse_gcs_uri

# Store cache: maps display_name -> (store_object, cached_at_timestamp)
# Stores rarely change, so 5-minute TTL is appropriate
_store_cache: Dict[str, Tuple[Any, float]] = {}
_store_list_cache: Tuple[Optional[List[Any]], float] = (None, 0.0)
STORE_CACHE_TTL_SECONDS = 300  # 5 minutes

# Get logger (config should be set by entry point)
logger = logging.getLogger(__name__)

# Suppress Gemini SDK info logs
logging.getLogger("google.genai").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Configuration from environment
STORE_NAME = os.getenv("FILE_STORE_NAME", "doc-intelligence-store")
SOURCE_DIRECTORY = os.getenv("SOURCE_DIRECTORY", "docs/structured")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
CHUNKING_CONFIG = {
    "white_space_config": {
        "max_tokens_per_chunk": int(os.getenv("MAX_TOKENS_PER_CHUNK", "512")),
        "max_overlap_tokens": int(os.getenv("MAX_OVERLAP_TOKENS", "100"))
    }
}

# Initialize client
client = genai.Client()

# Retry decorator for transient API failures
api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=32),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying {retry_state.fn.__name__} (attempt {retry_state.attempt_number}/3)..."
    )
)

def _log_event(
    event_type: str,
    file_name: str = None,
    details: dict = None,
    organization_id: str = None,
):
    """
    Log audit event asynchronously via centralized audit queue.

    Uses the shared audit queue to prevent connection pool proliferation.
    """
    try:
        from src.agents.core.audit_queue import enqueue_audit_event

        enqueue_audit_event(
            event_type=event_type,
            file_name=file_name,
            details=details,
            organization_id=organization_id,
        )
    except Exception as e:
        logger.warning(f"Failed to enqueue audit event: {e}")


def shutdown_audit_executor(wait: bool = True):
    """
    Shutdown the background audit queue.

    DEPRECATED: Use get_audit_queue().shutdown() instead.
    Kept for backwards compatibility.

    Args:
        wait: If True, wait for pending tasks to complete before returning.
    """
    try:
        from src.agents.core.audit_queue import get_audit_queue
        get_audit_queue().shutdown(wait=wait)
    except Exception as e:
        logger.warning(f"Error shutting down audit queue: {e}")


@api_retry
def create_file_search_store(display_name: str):
    """
    Create a new file search store.

    Args:
        display_name: Display name for the store

    Returns:
        FileSearchStore object
    """
    global _store_cache, _store_list_cache

    file_search_store = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    logger.info(f"Created store: {file_search_store.name}")

    # Update cache with new store
    _store_cache[display_name] = (file_search_store, time.time())
    # Invalidate list cache
    _store_list_cache = (None, 0.0)

    _log_event(
        "store_created",
        details={
            "store_name": file_search_store.name,
            "display_name": display_name,
        }
    )

    return file_search_store


def find_store_by_name(display_name: str):
    """
    Find a store by display name.

    Uses TTL-based cache to avoid repeated API calls.

    Args:
        display_name: Display name to search for

    Returns:
        FileSearchStore object or None if not found
    """
    global _store_cache

    # Check cache first
    if display_name in _store_cache:
        cached_store, cached_at = _store_cache[display_name]
        if time.time() - cached_at < STORE_CACHE_TTL_SECONDS:
            logger.debug(f"Store cache hit for {display_name}")
            return cached_store

    # Fetch from API and update cache
    for store in client.file_search_stores.list():
        # Cache all stores we see
        _store_cache[store.display_name] = (store, time.time())
        if store.display_name == display_name:
            logger.debug(f"Found and cached store {display_name}")
            return store

    # Store not found - cache the miss with None
    _store_cache[display_name] = (None, time.time())
    return None


def get_or_create_store(display_name: str):
    """
    Get existing store by display name or create a new one.

    Args:
        display_name: Display name to search for or create

    Returns:
        FileSearchStore object
    """
    existing = find_store_by_name(display_name)
    if existing:
        logger.info(f"Found existing store: {existing.name}")
        return existing
    return create_file_search_store(display_name)


def generate_store_display_name(org_name: str) -> str:
    """
    Generate store display name from organization name.

    Naming convention: <org_name>_file_search_store (all lowercase)
    Spaces and hyphens are replaced with underscores.

    Args:
        org_name: Organization name (e.g., "ACME Corp")

    Returns:
        Formatted display name (e.g., "acme_corp_file_search_store")
    """
    # Replace spaces and hyphens with underscores, convert to lowercase
    safe_name = org_name.lower().replace(" ", "_").replace("-", "_")
    # Remove any double underscores that might result
    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")
    return f"{safe_name}_file_search_store"


def get_or_create_store_by_org_name(org_name: str) -> tuple:
    """
    Get existing Gemini store by org name pattern or create a new one.

    Uses the naming convention: <org_name>_file_search_store

    Args:
        org_name: Organization name

    Returns:
        Tuple of (FileSearchStore, is_new: bool)
    """
    display_name = generate_store_display_name(org_name)

    # Try to find existing store
    existing = find_store_by_name(display_name)
    if existing:
        logger.info(f"Found existing Gemini store: {existing.name} ({display_name})")
        return existing, False

    # Create new store
    new_store = create_file_search_store(display_name)
    logger.info(f"Created new Gemini store: {new_store.name} ({display_name})")
    return new_store, True


def upload_file(
    file_search_store,
    file_path: str,
    organization_id: str = None,
    folder_id: str = None,
    folder_name: str = None,
    # Enhanced metadata for document traceability
    org_name: str = None,
    content_hash: str = None,
    original_gcs_path: str = None,
    parsed_gcs_path: str = None,
    original_file_extension: str = None,
    original_file_size: int = None,
    parse_date: str = None,
    parser_version: str = None,
):
    """
    Upload a single file to the file search store with comprehensive metadata.
    If document already exists, it will be deleted and re-uploaded.

    Args:
        file_search_store: The file search store object
        file_path: Path to the file to upload
        organization_id: Organization ID for multi-tenancy (optional, for metadata)
        folder_id: Folder ID for document organization (optional)
        folder_name: Folder name for query filtering (optional)
        org_name: Organization name for metadata (optional)
        content_hash: SHA-256 hash of file content for deduplication (optional)
        original_gcs_path: Original document GCS path before parsing (optional)
        parsed_gcs_path: Parsed document GCS path (optional)
        original_file_extension: Original file extension e.g. '.pdf' (optional)
        original_file_size: Original file size in bytes (optional)
        parse_date: ISO timestamp when document was parsed (optional)
        parser_version: Parser version used e.g. 'llama_parse_v2.5' (optional)
    """
    file_name = os.path.basename(file_path)
    replaced = False

    # Check if document already exists and delete it
    for doc in client.file_search_stores.documents.list(parent=file_search_store.name):
        if doc.display_name == file_name:
            logger.info(f"Document '{file_name}' already exists, replacing...")
            client.file_search_stores.documents.delete(name=doc.name, config={"force": True})
            replaced = True
            _log_event(
                "file_replaced",
                file_name=file_name,
                details={"store_name": file_search_store.name, "folder_name": folder_name}
            )
            break

    file_ext = os.path.splitext(file_name)[1].lstrip('.')

    # Handle GCS paths - download to temp file for Gemini upload
    temp_file_path = None
    if is_gcs_path(file_path):
        import tempfile
        from google.cloud import storage as gcs_storage
        gcs_client = gcs_storage.Client()
        # Parse: gs://bucket-name/path/to/file
        bucket_name, blob_name = parse_gcs_uri(file_path)
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.reload()  # Fetch metadata from GCS
        file_size = blob.size

        # Download to temp file (Gemini API requires local path)
        temp_file_path = tempfile.mktemp(suffix=os.path.splitext(file_name)[1])
        blob.download_to_filename(temp_file_path)
        upload_path = temp_file_path
    else:
        file_size = os.path.getsize(file_path)
        upload_path = file_path

    upload_date = datetime.datetime.now().isoformat()

    # Build custom metadata with folder information
    custom_metadata = [
        {"key": "file_name", "string_value": file_name},
        {"key": "document_type", "string_value": file_ext},
        {"key": "upload_date", "string_value": upload_date},
        {"key": "file_size", "numeric_value": file_size}
    ]

    # Add organization metadata if provided
    if organization_id:
        custom_metadata.append({"key": "organization_id", "string_value": organization_id})
    if org_name:
        custom_metadata.append({"key": "org_name", "string_value": org_name})

    # Add folder metadata if provided (enables folder-scoped queries)
    if folder_id:
        custom_metadata.append({"key": "folder_id", "string_value": folder_id})
    if folder_name:
        custom_metadata.append({"key": "folder_name", "string_value": folder_name})

    # Add enhanced metadata for document traceability
    if content_hash:
        custom_metadata.append({"key": "content_hash", "string_value": content_hash})
    if original_gcs_path:
        custom_metadata.append({"key": "original_gcs_path", "string_value": original_gcs_path})
    if parsed_gcs_path:
        custom_metadata.append({"key": "parsed_gcs_path", "string_value": parsed_gcs_path})
    if original_file_extension:
        custom_metadata.append({"key": "original_file_extension", "string_value": original_file_extension})
    if original_file_size is not None:
        custom_metadata.append({"key": "original_file_size", "numeric_value": original_file_size})
    if parse_date:
        custom_metadata.append({"key": "parse_date", "string_value": parse_date})
    if parser_version:
        custom_metadata.append({"key": "parser_version", "string_value": parser_version})

    operation = client.file_search_stores.upload_to_file_search_store(
        file_search_store_name=file_search_store.name,
        file=upload_path,
        config={
            "display_name": file_name,
            "chunking_config": CHUNKING_CONFIG,
            "custom_metadata": custom_metadata
        }
    )

    # Wait until upload is complete with exponential backoff
    wait_time = 0.5
    max_wait = 10
    while not operation.done:
        time.sleep(wait_time)
        logger.debug(f"Waiting for {file_name} to upload (next check in {wait_time:.1f}s)...")
        operation = client.operations.get(operation)
        wait_time = min(wait_time * 2, max_wait)

    logger.info(f"Uploaded: {file_name}" + (f" to folder '{folder_name}'" if folder_name else ""))

    # Clean up temp file if we downloaded from GCS
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
        logger.debug(f"Cleaned up temp file: {temp_file_path}")

    _log_event(
        "file_uploaded",
        file_name=file_name,
        details={
            "store_name": file_search_store.name,
            "file_size": file_size,
            "file_type": file_ext,
            "replaced": replaced,
            "organization_id": organization_id,
            "org_name": org_name,
            "folder_id": folder_id,
            "folder_name": folder_name,
            "content_hash": content_hash,
            "original_gcs_path": original_gcs_path,
            "parsed_gcs_path": parsed_gcs_path,
            "parser_version": parser_version,
        }
    )


def upload_all_files(file_search_store, directory: str, max_workers: int = 3):
    """
    Upload all files from a directory to the file search store concurrently.

    Args:
        file_search_store: The file search store object
        directory: Directory containing files to upload
        max_workers: Maximum concurrent uploads (default: 3)
    """
    file_pattern = os.path.join(directory, "*")
    files = [f for f in glob.glob(file_pattern) if os.path.isfile(f)]

    if not files:
        logger.warning(f"No files found in {directory}")
        return

    logger.info(f"Found {len(files)} files to upload (max {max_workers} concurrent)")

    def _upload_with_error_handling(file_path):
        try:
            upload_file(file_search_store, file_path)
            return file_path, None
        except Exception as e:
            logger.error(f"Error uploading {file_path}: {e}")
            return file_path, str(e)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_upload_with_error_handling, f): f for f in files}
        for future in as_completed(futures):
            file_path, error = future.result()
            if error:
                logger.warning(f"Failed: {os.path.basename(file_path)}")


@api_retry
def query_store(
    file_search_store,
    prompt: str,
    file_name_filter: str = None,
    folder_name_filter: str = None,
    folder_id_filter: str = None,
    search_mode: str = "semantic",
    generate_answer: bool = True,
    top_k: int = 5,
):
    """
    Query the file search store with optional folder/file filtering and search modes.

    Args:
        file_search_store: The file search store object
        prompt: The query prompt
        file_name_filter: Optional file name to filter by
        folder_name_filter: Optional folder name to filter by (for folder-scoped search)
        folder_id_filter: Optional folder ID to filter by
        search_mode: Search mode - 'semantic' (vector), 'keyword' (BM25), or 'hybrid' (combined)
        generate_answer: Whether to generate an answer from retrieved chunks
        top_k: Number of results to retrieve

    Returns:
        Response object from the model (or dict with citations if generate_answer=False)

    Search Scopes:
        - Org-wide: Leave all filters empty to search ALL indexed documents
        - Folder: Set folder_name_filter to search files in a specific folder
        - Single file: Set file_name_filter to search within a specific file

    Examples:
        # Org-wide search (cross-folder)
        query_store(store, "find all contracts")

        # Folder-scoped search
        query_store(store, "find invoices", folder_name_filter="Invoices 2024")

        # Single file search
        query_store(store, "summarize", file_name_filter="contract.pdf")

        # Hybrid search with keyword + semantic
        query_store(store, "payment terms NET 30", search_mode="hybrid")
    """
    file_search_kwargs = {"file_search_store_names": [file_search_store.name]}

    # Build metadata filter for folder/file scoping
    filters = []
    if file_name_filter:
        filters.append(f'file_name="{file_name_filter}"')
    if folder_name_filter:
        filters.append(f'folder_name="{folder_name_filter}"')
    if folder_id_filter:
        filters.append(f'folder_id="{folder_id_filter}"')

    if filters:
        # Combine filters with AND logic
        file_search_kwargs["metadata_filter"] = " AND ".join(filters)

    # Configure retrieval based on search mode
    # Gemini File Search API supports semantic retrieval by default
    # For hybrid/keyword modes, we enhance the prompt to leverage both
    enhanced_prompt = prompt
    if search_mode == "keyword":
        # For keyword mode, instruct to prioritize exact matches
        enhanced_prompt = f"Find documents containing these exact terms: {prompt}. Focus on keyword matches."
    elif search_mode == "hybrid":
        # For hybrid mode, combine semantic understanding with keyword matching
        enhanced_prompt = f"Search for: {prompt}. Consider both semantic meaning and exact keyword matches."

    file_search_config = types.FileSearch(**file_search_kwargs)

    if generate_answer:
        # Generate content with answer from retrieved chunks
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=enhanced_prompt,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(file_search=file_search_config)
                ]
            )
        )
    else:
        # Retrieval only - still need to call generate_content but with a retrieval-focused prompt
        retrieval_prompt = f"Retrieve the top {top_k} most relevant passages for: {enhanced_prompt}. Return only the citations."
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=retrieval_prompt,
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(file_search=file_search_config)
                ]
            )
        )

    # Log the search event
    has_results = bool(response.text) if response else False
    _log_event(
        "search_performed",
        file_name=file_name_filter,
        details={
            "query": prompt[:200] if len(prompt) > 200 else prompt,
            "store_name": file_search_store.name,
            "model": GEMINI_MODEL,
            "file_filter": file_name_filter,
            "folder_name_filter": folder_name_filter,
            "folder_id_filter": folder_id_filter,
            "search_mode": search_mode,
            "generate_answer": generate_answer,
            "top_k": top_k,
            "has_results": has_results,
        }
    )

    return response


def extract_citations(response):
    """
    Extract citations from a response.

    Args:
        response: Response object from generate_content

    Returns:
        List of citation dictionaries with text preview and title
    """
    citations = []

    if not response.candidates or not response.candidates[0].grounding_metadata:
        return citations

    grounding_metadata = response.candidates[0].grounding_metadata

    if grounding_metadata.grounding_chunks:
        for i, chunk in enumerate(grounding_metadata.grounding_chunks):
            citation = {
                "index": i + 1,
                "title": chunk.retrieved_context.title if chunk.retrieved_context else None,
                "text_preview": chunk.retrieved_context.text[:200] + "..."
                    if chunk.retrieved_context and len(chunk.retrieved_context.text) > 200
                    else chunk.retrieved_context.text if chunk.retrieved_context else None,
                "full_text": chunk.retrieved_context.text if chunk.retrieved_context else None
            }
            citations.append(citation)

    return citations


def list_documents(file_search_store):
    """
    List all documents in the file search store with their metadata.

    Args:
        file_search_store: The file search store object

    Returns:
        List of document objects
    """
    documents = []
    logger.info(f"Documents in {file_search_store.display_name}:")
    logger.info("=" * 60)

    for document in client.file_search_stores.documents.list(parent=file_search_store.name):
        documents.append(document)
        logger.info(f"  Name: {document.name}")
        logger.info(f"  Display Name: {document.display_name}")

        # Log all available attributes
        if hasattr(document, 'create_time') and document.create_time:
            logger.info(f"  Create Time: {document.create_time}")
        if hasattr(document, 'update_time') and document.update_time:
            logger.info(f"  Update Time: {document.update_time}")
        if hasattr(document, 'state') and document.state:
            logger.info(f"  State: {document.state}")
        if hasattr(document, 'size_bytes') and document.size_bytes:
            logger.info(f"  Size: {document.size_bytes} bytes")

        # Log custom metadata
        if hasattr(document, 'custom_metadata') and document.custom_metadata:
            logger.info(f"  Custom Metadata:")
            for meta in document.custom_metadata:
                if hasattr(meta, 'string_value') and meta.string_value:
                    logger.info(f"    - {meta.key}: {meta.string_value}")
                elif hasattr(meta, 'numeric_value') and meta.numeric_value is not None:
                    logger.info(f"    - {meta.key}: {meta.numeric_value}")

        logger.info("-" * 60)

    logger.info(f"Total documents: {len(documents)}")
    return documents


@api_retry
def delete_store(store_name: str):
    """
    Delete a file search store.

    Args:
        store_name: Full name of the store (e.g., 'fileSearchStores/xxx')
    """
    global _store_cache, _store_list_cache

    client.file_search_stores.delete(name=store_name, config={"force": True})
    logger.info(f"Deleted store: {store_name}")

    # Invalidate caches - clear all since we don't know the display name
    _store_cache.clear()
    _store_list_cache = (None, 0.0)

    _log_event(
        "store_deleted",
        details={"store_name": store_name}
    )


def list_all_stores():
    """
    List all file search stores.

    Uses TTL-based cache to avoid repeated API calls.

    Returns:
        List of store objects
    """
    global _store_cache, _store_list_cache

    # Check list cache first
    cached_stores, cached_at = _store_list_cache
    if cached_stores is not None and time.time() - cached_at < STORE_CACHE_TTL_SECONDS:
        logger.debug("Store list cache hit")
        return cached_stores

    stores = []
    logger.info("All File Search Stores:")
    logger.info("-" * 50)

    for store in client.file_search_stores.list():
        stores.append(store)
        # Update individual store cache
        _store_cache[store.display_name] = (store, time.time())
        logger.info(f"  Name: {store.name}")
        logger.info(f"  Display Name: {store.display_name}")
        logger.info(f"  Active Documents: {store.active_documents_count}")
        logger.info("-" * 50)

    # Cache the list
    _store_list_cache = (stores, time.time())
    logger.debug(f"Cached {len(stores)} stores")

    return stores
