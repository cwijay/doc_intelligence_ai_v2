"""
LangGraph StateGraph for document processing workflow.

Defines a stateful workflow for processing individual documents:
parse -> index -> generate -> finalize

Supports checkpointing for durability and recovery from failures.
"""

import asyncio
import hashlib
import logging
import time
from typing import TypedDict, Optional, List, Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.rag.gemini_file_store import upload_file, get_or_create_store_by_org_name
from src.db.repositories import bulk_repository
from src.services.parse_service import parse_and_save

from .config import get_bulk_config
from .schemas import DocumentItemStatus, ProcessingOptions

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITION
# =============================================================================


class DocumentState(TypedDict):
    """State for document processing workflow."""

    # Input (set once at start)
    document_id: str
    bulk_job_id: str
    original_path: str
    org_id: str
    org_name: str  # Human-readable org name for GCS paths
    folder_name: str
    options: dict  # ProcessingOptions as dict

    # Processing state (updated by nodes)
    status: str
    parsed_content: Optional[str]
    parsed_path: Optional[str]
    content_hash: Optional[str]
    indexed: bool
    summary: Optional[str]
    faqs: Optional[List[dict]]
    questions: Optional[List[dict]]

    # Metrics
    parse_time_ms: int
    index_time_ms: int
    generation_time_ms: int
    total_time_ms: int
    token_usage: int
    llamaparse_pages: int

    # Error handling
    error: Optional[str]
    retry_count: int


# =============================================================================
# NODE FUNCTIONS
# =============================================================================


async def is_document_cancelled(document_id: str) -> bool:
    """Check if a document has been cancelled (status = skipped)."""
    doc = await bulk_repository.get_document_item(document_id)
    if not doc:
        return True  # Document doesn't exist, treat as cancelled
    return doc.get("status") == DocumentItemStatus.SKIPPED.value


async def parse_node(state: DocumentState) -> DocumentState:
    """Parse the document using shared parse service."""
    logger.info(f"Parsing document: {state['original_path']}")
    start_time = time.time()

    try:
        # Update status in database
        await bulk_repository.update_document_item(
            state["document_id"],
            status=DocumentItemStatus.PARSING.value,
        )

        # Use shared parse service (handles GCS download, parsing, save, DB registration)
        result = await parse_and_save(
            file_path=state["original_path"],
            folder_name=state["folder_name"],
            org_id=state["org_id"],
            org_name=state["org_name"],
            save_to_gcs=True,
            check_cache=True,  # Skip re-parsing already parsed docs
        )

        if not result.success:
            return {
                **state,
                "status": DocumentItemStatus.FAILED.value,
                "error": result.error or "Parse failed",
                "parse_time_ms": result.parse_time_ms,
            }

        # Calculate content hash
        content_hash = hashlib.sha256(result.parsed_content.encode()).hexdigest()

        logger.info(f"Parsed document in {result.parse_time_ms}ms: {result.parsed_path}")

        return {
            **state,
            "status": DocumentItemStatus.PARSED.value,
            "parsed_content": result.parsed_content,
            "parsed_path": result.parsed_path,
            "content_hash": content_hash,
            "parse_time_ms": result.parse_time_ms,
            "llamaparse_pages": result.estimated_pages,
        }

    except Exception as e:
        parse_time = int((time.time() - start_time) * 1000)
        logger.error(f"Parse error for {state['original_path']}: {e}")
        return {
            **state,
            "status": DocumentItemStatus.FAILED.value,
            "error": f"Parse error: {str(e)}",
            "parse_time_ms": parse_time,
        }


async def index_node(state: DocumentState) -> DocumentState:
    """Index the parsed document to Gemini File Search."""
    if state.get("error"):
        return state

    # Check if job was cancelled before starting
    if await is_document_cancelled(state["document_id"]):
        logger.info(f"Document {state['document_id']} was cancelled, skipping indexing")
        return {
            **state,
            "status": DocumentItemStatus.SKIPPED.value,
            "error": "Job cancelled",
        }

    logger.info(f"Indexing document: {state['parsed_path']}")
    start_time = time.time()

    try:
        # Update status in database
        await bulk_repository.update_document_item(
            state["document_id"],
            status=DocumentItemStatus.INDEXING.value,
        )

        config = get_bulk_config()

        # Get or create file search store for org
        # Note: get_or_create_store_by_org_name is sync and returns (store, is_new) tuple
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: get_or_create_store_by_org_name(state["org_name"])
        )
        store, is_new = result  # Unpack the tuple

        if not store:
            return {
                **state,
                "status": DocumentItemStatus.FAILED.value,
                "error": "Failed to get/create file search store",
            }

        # Upload to Gemini File Search
        # Note: upload_file is sync, run in executor with timeout
        def do_upload():
            upload_file(
                store,
                file_path=state["parsed_path"],
                organization_id=state["org_id"],
                folder_name=state["folder_name"],
                org_name=state["org_name"],
                original_gcs_path=state["original_path"],
                parsed_gcs_path=state["parsed_path"],
            )

        await asyncio.wait_for(
            loop.run_in_executor(None, do_upload),
            timeout=config.index_timeout_seconds,
        )

        index_time = int((time.time() - start_time) * 1000)
        logger.info(f"Indexed document in {index_time}ms")

        return {
            **state,
            "status": DocumentItemStatus.INDEXED.value,
            "indexed": True,
            "index_time_ms": index_time,
        }

    except asyncio.TimeoutError:
        index_time = int((time.time() - start_time) * 1000)
        logger.error(f"Index timeout for {state['parsed_path']}")
        return {
            **state,
            "status": DocumentItemStatus.FAILED.value,
            "error": f"Index timeout after {config.index_timeout_seconds}s",
            "index_time_ms": index_time,
        }
    except Exception as e:
        index_time = int((time.time() - start_time) * 1000)
        logger.error(f"Index error for {state['parsed_path']}: {e}")
        return {
            **state,
            "status": DocumentItemStatus.FAILED.value,
            "error": f"Index error: {str(e)}",
            "index_time_ms": index_time,
        }


async def generate_node(state: DocumentState) -> DocumentState:
    """Generate content (summary, FAQs, questions) for the document."""
    if state.get("error"):
        return state

    # Check if job was cancelled before starting
    if await is_document_cancelled(state["document_id"]):
        logger.info(f"Document {state['document_id']} was cancelled, skipping generation")
        return {
            **state,
            "status": DocumentItemStatus.SKIPPED.value,
            "error": "Job cancelled",
        }

    options = ProcessingOptions(**state.get("options", {}))

    # Check if any generation is requested
    if not (options.generate_summary or options.generate_faqs or options.generate_questions):
        logger.info("No generation requested, skipping")
        return {
            **state,
            "status": DocumentItemStatus.COMPLETED.value,
        }

    logger.info(f"Generating content for: {state['parsed_path']}")
    start_time = time.time()

    try:
        # Update status in database
        await bulk_repository.update_document_item(
            state["document_id"],
            status=DocumentItemStatus.GENERATING.value,
        )

        config = get_bulk_config()

        # Import DocumentAgent here to avoid circular imports
        from src.agents.document.core import DocumentAgent
        from src.agents.document.schemas import GenerationOptions

        agent = DocumentAgent()

        # Create generation options
        gen_options = GenerationOptions(
            generate_summary=options.generate_summary,
            generate_faqs=options.generate_faqs,
            generate_questions=options.generate_questions,
            num_faqs=options.num_faqs,
            num_questions=options.num_questions,
            summary_max_words=options.summary_max_words,
        )

        # Generate content
        filename = state["original_path"].split("/")[-1]
        base_name = filename.rsplit(".", 1)[0]

        result = await asyncio.wait_for(
            agent.generate_all(
                document_name=base_name,
                parsed_file_path=state["parsed_path"],
                options=gen_options,
                organization_id=state["org_id"],
            ),
            timeout=config.generation_timeout_seconds,
        )

        generation_time = int((time.time() - start_time) * 1000)

        # Extract results
        summary = result.summary if result else None
        faqs = [faq.model_dump() for faq in result.faqs] if result and result.faqs else None
        questions = [q.model_dump() for q in result.questions] if result and result.questions else None

        # Estimate token usage (rough estimate)
        token_usage = 0
        if summary:
            token_usage += len(summary.split()) * 2
        if faqs:
            token_usage += len(faqs) * 100
        if questions:
            token_usage += len(questions) * 100

        logger.info(f"Generated content in {generation_time}ms")

        return {
            **state,
            "status": DocumentItemStatus.COMPLETED.value,
            "summary": summary,
            "faqs": faqs,
            "questions": questions,
            "generation_time_ms": generation_time,
            "token_usage": token_usage,
        }

    except asyncio.TimeoutError:
        generation_time = int((time.time() - start_time) * 1000)
        logger.error(f"Generation timeout for {state['parsed_path']}")
        return {
            **state,
            "status": DocumentItemStatus.FAILED.value,
            "error": f"Generation timeout after {config.generation_timeout_seconds}s",
            "generation_time_ms": generation_time,
        }
    except Exception as e:
        generation_time = int((time.time() - start_time) * 1000)
        logger.error(f"Generation error for {state['parsed_path']}: {e}")
        return {
            **state,
            "status": DocumentItemStatus.FAILED.value,
            "error": f"Generation error: {str(e)}",
            "generation_time_ms": generation_time,
        }


async def finalize_node(state: DocumentState) -> DocumentState:
    """Finalize document processing and update database."""
    total_time = (
        state.get("parse_time_ms", 0) +
        state.get("index_time_ms", 0) +
        state.get("generation_time_ms", 0)
    )

    # Update document item in database
    await bulk_repository.update_document_item(
        state["document_id"],
        status=state.get("status", DocumentItemStatus.COMPLETED.value),
        parsed_path=state.get("parsed_path"),
        error_message=state.get("error"),
        parse_time_ms=state.get("parse_time_ms"),
        index_time_ms=state.get("index_time_ms"),
        generation_time_ms=state.get("generation_time_ms"),
        total_time_ms=total_time,
        token_usage=state.get("token_usage", 0),
        llamaparse_pages=state.get("llamaparse_pages", 0),
        content_hash=state.get("content_hash"),
    )

    logger.info(
        f"Finalized document {state['document_id']}: "
        f"status={state.get('status')}, total_time={total_time}ms"
    )

    return {
        **state,
        "total_time_ms": total_time,
    }


async def error_handler_node(state: DocumentState) -> DocumentState:
    """Handle errors and update database."""
    logger.error(f"Error handler for document {state['document_id']}: {state.get('error')}")

    # Update document item with error
    await bulk_repository.update_document_item(
        state["document_id"],
        status=DocumentItemStatus.FAILED.value,
        error_message=state.get("error"),
        parse_time_ms=state.get("parse_time_ms"),
        index_time_ms=state.get("index_time_ms"),
        generation_time_ms=state.get("generation_time_ms"),
    )

    return state


# =============================================================================
# CONDITIONAL EDGES
# =============================================================================


def should_continue_after_parse(state: DocumentState) -> Literal["continue", "error"]:
    """Decide whether to continue after parsing."""
    if state.get("error"):
        return "error"
    if state.get("status") == DocumentItemStatus.PARSED.value:
        return "continue"
    return "error"


def should_continue_after_index(state: DocumentState) -> Literal["continue", "error"]:
    """Decide whether to continue after indexing."""
    if state.get("error"):
        return "error"
    if state.get("indexed"):
        return "continue"
    return "error"


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def create_document_processing_graph(use_memory_saver: bool = True):
    """
    Create the LangGraph for document processing.

    Args:
        use_memory_saver: Use in-memory checkpointer (default True).
                         Set False for production with PostgresSaver.

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(DocumentState)

    # Add nodes
    graph.add_node("parse", parse_node)
    graph.add_node("index", index_node)
    graph.add_node("generate", generate_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("error_handler", error_handler_node)

    # Add edges
    graph.add_conditional_edges(
        "parse",
        should_continue_after_parse,
        {
            "continue": "index",
            "error": "error_handler",
        }
    )

    graph.add_conditional_edges(
        "index",
        should_continue_after_index,
        {
            "continue": "generate",
            "error": "error_handler",
        }
    )

    graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("error_handler", END)

    # Set entry point
    graph.set_entry_point("parse")

    # Compile with checkpointer
    if use_memory_saver:
        checkpointer = MemorySaver()
        return graph.compile(checkpointer=checkpointer)
    else:
        # For production, use PostgresSaver (requires setup)
        return graph.compile()


def create_initial_state(
    document_id: str,
    bulk_job_id: str,
    original_path: str,
    org_id: str,
    org_name: str,
    folder_name: str,
    options: ProcessingOptions,
) -> DocumentState:
    """
    Create initial state for document processing.

    Args:
        document_id: Document item ID
        bulk_job_id: Parent bulk job ID
        original_path: GCS path to original document
        org_id: Organization ID
        org_name: Organization name (human-readable, for GCS paths)
        folder_name: Bulk folder name
        options: Processing options

    Returns:
        Initial DocumentState
    """
    return DocumentState(
        document_id=document_id,
        bulk_job_id=bulk_job_id,
        original_path=original_path,
        org_id=org_id,
        org_name=org_name,
        folder_name=folder_name,
        options=options.model_dump(),
        status=DocumentItemStatus.PENDING.value,
        parsed_content=None,
        parsed_path=None,
        content_hash=None,
        indexed=False,
        summary=None,
        faqs=None,
        questions=None,
        parse_time_ms=0,
        index_time_ms=0,
        generation_time_ms=0,
        total_time_ms=0,
        token_usage=0,
        llamaparse_pages=0,
        error=None,
        retry_count=0,
    )


# Singleton graph instance
_document_graph = None


def get_document_graph():
    """Get the document processing graph singleton."""
    global _document_graph
    if _document_graph is None:
        config = get_bulk_config()
        _document_graph = create_document_processing_graph(
            use_memory_saver=not config.use_postgres_checkpointer
        )
    return _document_graph


def reset_document_graph():
    """Reset the graph singleton (for testing)."""
    global _document_graph
    _document_graph = None
