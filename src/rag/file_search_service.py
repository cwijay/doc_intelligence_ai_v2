"""
File Search Service - High-level facade for Gemini File Search operations.

This module provides orchestration functions that combine low-level
API operations from gemini_file_store.py and llamaindex_utils.
"""

import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from src.utils.timer_utils import elapsed_ms

load_dotenv()

from src.rag.gemini_file_store import (
    STORE_NAME,
    SOURCE_DIRECTORY,
    get_or_create_store,
    upload_all_files,
    list_documents,
    query_store,
    extract_citations,
)
from src.rag.llama_parse_util import parse_document
from src.db.repositories.audit_repository import (
    register_document,
    find_cached_result,
    start_job,
    complete_job,
    fail_job,
    log_event,
)

# Output directory for parsed documents (from storage config)
def _get_parsed_dir():
    """Get parsed directory from storage config."""
    from src.storage import get_storage_config
    return get_storage_config().parsed_directory

PARSED_OUTPUT_DIR = os.getenv("PARSED_OUTPUT_DIR", "parsed")  # Fallback default

# Parsing model configuration
PARSING_MODEL_HIGH_COMPLEXITY = os.getenv("PARSING_MODEL_HIGH", "gemini-2.5-pro")
PARSING_MODEL_STANDARD = os.getenv("PARSING_MODEL_STANDARD", "openai-gpt-5-mini")

logger = logging.getLogger(__name__)


def _log_search_result(response, context: str):
    """Helper to log search response and citations."""
    logger.info("=" * 60)
    logger.info(f"SEARCH: {context}")
    logger.info("=" * 60)
    logger.info(f"Answer: {response.text}")

    citations = extract_citations(response)
    if citations:
        logger.info("Citations:")
        for c in citations:
            logger.info(f"  [{c['title']}]")
            logger.info(f"  {c['text_preview']}")


def setup_store():
    """
    Create store and upload documents.
    Call this function manually when you need to ingest/update documents.

    Returns:
        The file search store object
    """
    store = get_or_create_store(STORE_NAME)
    logger.info(f"Uploading files from {SOURCE_DIRECTORY}...")
    upload_all_files(store, SOURCE_DIRECTORY)
    list_documents(store)
    return store


def search_by_document(store, query: str, file_name: str):
    """
    Search in a specific document.

    Args:
        store: The file search store
        query: The search query
        file_name: Name of the document to search in

    Returns:
        Response object from the model
    """
    response = query_store(store, query, file_name_filter=file_name)
    _log_search_result(response, f"{file_name} only")
    return response


def search_all_documents(store, query: str):
    """
    Search across all documents in the store.

    Args:
        store: The file search store
        query: The search query

    Returns:
        Response object from the model
    """
    response = query_store(store, query)
    _log_search_result(response, "All documents")
    return response


def parse_handwritten_document(
    file_path: str,
    output_dir: str = PARSED_OUTPUT_DIR,
    use_agent_mode: bool = True,
    complexity: str = "normal",
    skip_cache: bool = False
) -> str:
    """
    Parse a handwritten document using LlamaParse and save as markdown.

    Features caching and audit logging via PostgreSQL:
    - Avoids re-parsing identical documents with same model
    - Tracks processing history and job status
    - Logs all operations for auditability

    Args:
        file_path: Path to the document to parse
        output_dir: Output directory for markdown file (default: parsed/)
        use_agent_mode: Use agent parsing mode for better accuracy (default: True)
        complexity: Document complexity ("normal" or "high")
                   - "normal": openai-gpt-5-mini (default, cost-effective)
                   - "high": gemini-2.5-pro (better for complex handwriting)
        skip_cache: Force re-processing even if cached result exists (default: False)

    Returns:
        str: Parsed markdown content

    Example:
        >>> content = parse_handwritten_document("simple.pdf")
        >>> content = parse_handwritten_document("complex.pdf", complexity="high")
        >>> content = parse_handwritten_document("doc.pdf", skip_cache=True)
    """
    logger.info("=" * 60)
    logger.info(f"PARSING: {file_path} (complexity={complexity})")
    logger.info("=" * 60)

    # Register document and get hash
    file_hash = register_document(file_path)
    file_name = Path(file_path).name
    model = PARSING_MODEL_HIGH_COMPLEXITY if complexity == "high" else PARSING_MODEL_STANDARD

    # Check cache
    if not skip_cache:
        cached_path = find_cached_result(file_hash, model)
        if cached_path:
            log_event("cache_hit", document_hash=file_hash, file_name=file_name)
            logger.info(f"Cache hit for {file_path}")
            try:
                # Read from GCS storage
                import asyncio
                from src.storage import get_storage

                async def read_cached():
                    storage = get_storage()
                    return await storage.read(cached_path)

                try:
                    loop = asyncio.get_running_loop()
                    import nest_asyncio
                    nest_asyncio.apply()
                    content = loop.run_until_complete(read_cached())
                except RuntimeError:
                    content = asyncio.run(read_cached())

                if content:
                    return content
                logger.warning(f"Cached file not found in GCS: {cached_path}, re-processing...")
            except Exception as e:
                logger.warning(f"Failed to read cached file from GCS: {e}, re-processing...")

    # Start job
    job_id = start_job(file_hash, file_name, model, complexity)
    log_event("parse_started", document_hash=file_hash, file_name=file_name, job_id=job_id)
    start_time = time.time()

    try:
        content = parse_document(
            file_path,
            output_dir=output_dir,
            use_agent_mode=use_agent_mode,
            complexity=complexity
        )

        output_path = f"{output_dir}/{Path(file_path).stem}.md"
        duration_ms = int(elapsed_ms(start_time))

        complete_job(job_id, output_path, duration_ms)
        log_event(
            "parse_completed",
            document_hash=file_hash,
            file_name=file_name,
            job_id=job_id,
            details={"duration_ms": duration_ms}
        )

        logger.info(f"Successfully parsed document to {output_dir}/ ({duration_ms}ms)")
        return content

    except Exception as e:
        fail_job(job_id, str(e))
        log_event(
            "error",
            document_hash=file_hash,
            file_name=file_name,
            job_id=job_id,
            details={"error": str(e)}
        )
        raise
