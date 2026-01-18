"""RAG search tool using Gemini File Search with semantic caching."""

import json
import logging
import time
from typing import Optional, Type, List

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import RAGSearchInput
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class RAGSearchTool(BaseTool):
    """Tool to search documents using Gemini File Search with conversation context."""

    name: str = "rag_search"
    description: str = """Search uploaded documents and answer questions using RAG.
    Supports semantic, keyword, and hybrid search modes.
    Can filter by folder or specific file.
    Returns answer with citations from source documents.
    Use this tool when the user asks questions about their documents."""
    args_schema: Type[BaseModel] = RAGSearchInput

    config: DocumentAgentConfig = Field(default_factory=DocumentAgentConfig)

    # Bound filter values from request context - used when LLM doesn't extract filters
    # These are set via with_bound_filters() to ensure correct cache scoping
    # Note: Using regular fields (not underscore-prefixed) because Pydantic v2 ignores
    # underscore-prefixed kwargs in constructor, causing bound filters to never be set
    bound_file_filter: Optional[List[str]] = Field(default=None, exclude=True)
    bound_folder_filter: Optional[str] = Field(default=None, exclude=True)

    def with_bound_filters(
        self,
        file_filter: Optional[List[str]] = None,
        folder_filter: Optional[str] = None,
    ) -> "RAGSearchTool":
        """Create a new tool instance with bound filter values.

        Bound filters are used for cache operations when the LLM doesn't
        extract filter values from the query text. This ensures the semantic
        cache correctly scopes queries by document.

        Args:
            file_filter: List of file names to scope cache operations
            folder_filter: Folder name to scope cache operations

        Returns:
            New RAGSearchTool instance with bound filters
        """
        return RAGSearchTool(
            config=self.config,
            bound_file_filter=file_filter,
            bound_folder_filter=folder_filter,
        )

    def _run(
        self,
        query: str,
        organization_name: str,
        folder_filter: Optional[str] = None,
        file_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        max_sources: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Search documents using Gemini File Search with semantic caching."""
        start_time = time.time()

        # Resolve filter values: BOUND filters (from API request) take precedence over LLM-provided values
        # This prevents the LLM from using bogus values like "rag_search" as file names
        if self.bound_file_filter:
            # Use authoritative bound list of file names from API request
            effective_file_filter = self.bound_file_filter
        elif file_filter:
            # Fallback: LLM provided a single file name - wrap in list
            effective_file_filter = [file_filter]
        else:
            effective_file_filter = None

        effective_folder_filter = self.bound_folder_filter if self.bound_folder_filter else folder_filter

        if self.bound_file_filter or self.bound_folder_filter:
            logger.debug(
                f"Using bound filters (from API) - file: {effective_file_filter}, folder: {effective_folder_filter} "
                f"(LLM provided: file={file_filter!r}, folder={folder_filter!r})"
            )

        # Log effective filters for debugging
        logger.info(f"RAG search filters - file: {effective_file_filter}, folder: {effective_folder_filter}")

        try:
            from src.rag.gemini_file_store import query_store, generate_store_display_name, get_client
            from src.db.repositories import rag_repository
            from src.db.repositories import semantic_cache_repository
            from src.rag.embeddings import get_query_embedding_sync

            # Generate normalized store name for DB lookup
            store_name = generate_store_display_name(organization_name)

            async def do_search():
                # Look up store from PostgreSQL (faster than Gemini API listing)
                store_record = await rag_repository.get_store_by_display_name(store_name)
                if not store_record:
                    return {
                        "success": False,
                        "error": f"File search store not found for organization: {organization_name}. Please ensure documents have been ingested first.",
                        "query": query
                    }

                org_id = store_record.get("organization_id")

                # =============================================================
                # SEMANTIC CACHE: Check for similar cached queries
                # =============================================================
                if semantic_cache_repository.is_cache_enabled() and org_id:
                    try:
                        # Generate embedding for the query
                        query_embedding = get_query_embedding_sync(query)

                        # Look for semantically similar cached query
                        cached = await semantic_cache_repository.find_similar_query(
                            org_id=org_id,
                            query_embedding=query_embedding,
                            folder_filter=effective_folder_filter,
                            file_filter=effective_file_filter,
                        )

                        if cached:
                            logger.info(
                                f"Semantic cache HIT: similarity={cached['similarity']:.2f}, "
                                f"returning cached answer"
                            )
                            return {
                                "success": True,
                                "answer": cached["answer"],
                                "citations": cached["citations"],
                                "query": query,
                                "search_mode": cached.get("search_mode", search_mode),
                                "filters": {
                                    "folder": effective_folder_filter,
                                    "file": effective_file_filter
                                },
                                "cached": True,
                                "cache_similarity": cached["similarity"],
                            }
                    except Exception as cache_error:
                        logger.warning(f"Semantic cache lookup failed: {cache_error}")
                        query_embedding = None
                else:
                    query_embedding = None

                # =============================================================
                # CACHE MISS: Perform Gemini File Search
                # =============================================================

                # Get Gemini store object using the stored gemini_store_id
                gemini_store_id = store_record.get("gemini_store_id")
                if not gemini_store_id:
                    return {
                        "success": False,
                        "error": f"Gemini store ID not found in database for organization: {organization_name}",
                        "query": query
                    }

                store = get_client().file_search_stores.get(name=gemini_store_id)
                if not store:
                    return {
                        "success": False,
                        "error": f"Gemini store not found: {gemini_store_id}",
                        "query": query
                    }

                # Perform the search
                response = query_store(
                    file_search_store=store,
                    prompt=query,
                    file_name_filter=effective_file_filter,
                    folder_name_filter=effective_folder_filter,
                    search_mode=search_mode,
                    generate_answer=True,
                    top_k=max_sources
                )

                # Extract answer and citations from response
                answer = ""
                citations = []

                if hasattr(response, 'text'):
                    answer = response.text
                elif hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and candidate.content.parts:
                        answer = candidate.content.parts[0].text

                # Extract grounding metadata for citations
                avg_confidence = 0.0
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata'):
                        grounding = candidate.grounding_metadata

                        # Debug: log grounding metadata structure
                        logger.info(f"Grounding metadata attributes: {[a for a in dir(grounding) if not a.startswith('_')]}")

                        # Extract average confidence from grounding_supports if available
                        if hasattr(grounding, 'grounding_supports') and grounding.grounding_supports:
                            confidence_scores = []
                            for support in grounding.grounding_supports:
                                if hasattr(support, 'confidence_scores') and support.confidence_scores:
                                    confidence_scores.extend(support.confidence_scores)
                            if confidence_scores:
                                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                                logger.info(f"Extracted avg confidence from grounding_supports: {avg_confidence:.3f}")

                        if hasattr(grounding, 'grounding_chunks'):
                            for i, chunk in enumerate(grounding.grounding_chunks):
                                # Debug: log chunk attributes for first chunk
                                if i == 0:
                                    chunk_attrs = [a for a in dir(chunk) if not a.startswith('_')]
                                    logger.info(f"Grounding chunk[0] attributes: {chunk_attrs}")

                                # Use retrieved_context for text and title (Gemini File Search structure)
                                ctx = chunk.retrieved_context if hasattr(chunk, 'retrieved_context') else None
                                if i == 0 and ctx:
                                    ctx_attrs = [a for a in dir(ctx) if not a.startswith('_')]
                                    logger.info(f"retrieved_context attributes: {ctx_attrs}")
                                elif i == 0:
                                    logger.info(f"No retrieved_context on chunk, raw chunk: {chunk}")

                                # Calculate relevance score: use confidence if available, else default to 0.8
                                relevance = avg_confidence if avg_confidence > 0 else (0.8 if ctx else 0.0)

                                citation = {
                                    "text": ctx.text if ctx and hasattr(ctx, 'text') else "",
                                    "file": ctx.title if ctx and hasattr(ctx, 'title') else "",
                                    "relevance_score": relevance
                                }
                                citations.append(citation)
                                logger.info(f"Citation {i}: file='{citation['file']}', text_len={len(citation['text'])}, relevance={relevance:.3f}")

                # =============================================================
                # CACHE STORE: Save result for future similar queries
                # =============================================================
                if semantic_cache_repository.is_cache_enabled() and org_id and answer:
                    try:
                        # Generate embedding if we didn't already (cache was disabled earlier)
                        if query_embedding is None:
                            query_embedding = get_query_embedding_sync(query)

                        await semantic_cache_repository.cache_query(
                            org_id=org_id,
                            query_text=query,
                            query_embedding=query_embedding,
                            answer=answer,
                            citations=citations,
                            folder_filter=effective_folder_filter,
                            file_filter=effective_file_filter,
                            search_mode=search_mode,
                        )
                        logger.debug(f"Cached RAG response for future similar queries (file_filter={effective_file_filter})")
                    except Exception as cache_error:
                        logger.warning(f"Failed to cache RAG response: {cache_error}")

                return {
                    "success": True,
                    "answer": answer,
                    "citations": citations,
                    "query": query,
                    "search_mode": search_mode,
                    "filters": {
                        "folder": effective_folder_filter,
                        "file": effective_file_filter
                    },
                    "cached": False,
                }

            result = run_async(do_search())
            duration_ms = elapsed_ms(start_time)
            result["processing_time_ms"] = duration_ms

            cached_str = " (CACHED)" if result.get("cached") else ""
            logger.info(f"RAG search completed{cached_str}: {len(result.get('citations', []))} citations in {duration_ms:.1f}ms")
            return json.dumps(result)

        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e),
                "query": query
            })

    async def _arun(
        self,
        query: str,
        organization_name: str,
        folder_filter: Optional[str] = None,
        file_filter: Optional[str] = None,
        search_mode: str = "hybrid",
        max_sources: int = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Async version: Search documents using Gemini File Search with semantic caching."""
        start_time = time.time()

        # Resolve filter values: BOUND filters (from API request) take precedence over LLM-provided values
        # This prevents the LLM from using bogus values like "rag_search" as file names
        if self.bound_file_filter:
            # Use authoritative bound list of file names from API request
            effective_file_filter = self.bound_file_filter
        elif file_filter:
            # Fallback: LLM provided a single file name - wrap in list
            effective_file_filter = [file_filter]
        else:
            effective_file_filter = None

        effective_folder_filter = self.bound_folder_filter if self.bound_folder_filter else folder_filter

        if self.bound_file_filter or self.bound_folder_filter:
            logger.debug(
                f"Using bound filters (from API) - file: {effective_file_filter}, folder: {effective_folder_filter} "
                f"(LLM provided: file={file_filter!r}, folder={folder_filter!r})"
            )

        # Log effective filters for debugging
        logger.info(f"RAG search filters - file: {effective_file_filter}, folder: {effective_folder_filter}")

        try:
            import asyncio
            from src.rag.gemini_file_store import query_store, generate_store_display_name, get_client
            from src.db.repositories import rag_repository
            from src.db.repositories import semantic_cache_repository
            from src.rag.embeddings import get_query_embedding_sync

            # Generate normalized store name for DB lookup
            store_name = generate_store_display_name(organization_name)

            # =============================================================
            # PERF: Parallelize store lookup + embedding generation
            # This saves ~1.0s by running both operations concurrently
            # =============================================================
            async def get_embedding_async():
                """Run sync embedding in executor to not block event loop."""
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, get_query_embedding_sync, query)

            # Run store lookup and embedding generation in parallel
            cache_enabled = semantic_cache_repository.is_cache_enabled()
            if cache_enabled:
                store_record, query_embedding = await asyncio.gather(
                    rag_repository.get_store_by_display_name(store_name),
                    get_embedding_async()
                )
            else:
                # Cache disabled - just do store lookup
                store_record = await rag_repository.get_store_by_display_name(store_name)
                query_embedding = None

            if not store_record:
                return json.dumps({
                    "success": False,
                    "error": f"File search store not found for organization: {organization_name}. Please ensure documents have been ingested first.",
                    "query": query
                })

            org_id = store_record.get("organization_id")

            # =============================================================
            # SEMANTIC CACHE: Check for similar cached queries
            # =============================================================
            if cache_enabled and org_id and query_embedding is not None:
                try:
                    # Look for semantically similar cached query
                    cached = await semantic_cache_repository.find_similar_query(
                        org_id=org_id,
                        query_embedding=query_embedding,
                        folder_filter=effective_folder_filter,
                        file_filter=effective_file_filter,
                    )

                    if cached:
                        logger.info(
                            f"Semantic cache HIT: similarity={cached['similarity']:.2f}, "
                            f"returning cached answer"
                        )
                        result = {
                            "success": True,
                            "answer": cached["answer"],
                            "citations": cached["citations"],
                            "query": query,
                            "search_mode": cached.get("search_mode", search_mode),
                            "filters": {
                                "folder": effective_folder_filter,
                                "file": effective_file_filter
                            },
                            "cached": True,
                            "cache_similarity": cached["similarity"],
                            "processing_time_ms": elapsed_ms(start_time),
                        }
                        logger.info(f"RAG search completed (CACHED): {len(result.get('citations', []))} citations in {result['processing_time_ms']:.1f}ms")
                        return json.dumps(result)
                except Exception as cache_error:
                    logger.warning(f"Semantic cache lookup failed: {cache_error}")

            # =============================================================
            # CACHE MISS: Perform Gemini File Search
            # =============================================================

            # Get Gemini store object using the stored gemini_store_id
            gemini_store_id = store_record.get("gemini_store_id")
            if not gemini_store_id:
                return json.dumps({
                    "success": False,
                    "error": f"Gemini store ID not found in database for organization: {organization_name}",
                    "query": query
                })

            store = get_client().file_search_stores.get(name=gemini_store_id)
            if not store:
                return json.dumps({
                    "success": False,
                    "error": f"Gemini store not found: {gemini_store_id}",
                    "query": query
                })

            # Perform the search (sync Gemini API call)
            response = query_store(
                file_search_store=store,
                prompt=query,
                file_name_filter=effective_file_filter,
                folder_name_filter=effective_folder_filter,
                search_mode=search_mode,
                generate_answer=True,
                top_k=max_sources
            )

            # Extract answer and citations from response
            answer = ""
            citations = []

            if hasattr(response, 'text'):
                answer = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    answer = candidate.content.parts[0].text

            # Extract grounding metadata for citations
            avg_confidence = 0.0
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata

                    # Debug: log grounding metadata structure
                    logger.info(f"Grounding metadata attributes: {[a for a in dir(grounding) if not a.startswith('_')]}")

                    # Extract average confidence from grounding_supports if available
                    if hasattr(grounding, 'grounding_supports') and grounding.grounding_supports:
                        confidence_scores = []
                        for support in grounding.grounding_supports:
                            if hasattr(support, 'confidence_scores') and support.confidence_scores:
                                confidence_scores.extend(support.confidence_scores)
                        if confidence_scores:
                            avg_confidence = sum(confidence_scores) / len(confidence_scores)
                            logger.info(f"Extracted avg confidence from grounding_supports: {avg_confidence:.3f}")

                    if hasattr(grounding, 'grounding_chunks'):
                        for i, chunk in enumerate(grounding.grounding_chunks):
                            # Debug: log chunk attributes for first chunk
                            if i == 0:
                                chunk_attrs = [a for a in dir(chunk) if not a.startswith('_')]
                                logger.info(f"Grounding chunk[0] attributes: {chunk_attrs}")

                            # Use retrieved_context for text and title (Gemini File Search structure)
                            ctx = chunk.retrieved_context if hasattr(chunk, 'retrieved_context') else None
                            if i == 0 and ctx:
                                ctx_attrs = [a for a in dir(ctx) if not a.startswith('_')]
                                logger.info(f"retrieved_context attributes: {ctx_attrs}")
                            elif i == 0:
                                logger.info(f"No retrieved_context on chunk, raw chunk: {chunk}")

                            # Calculate relevance score: use confidence if available, else default to 0.8
                            relevance = avg_confidence if avg_confidence > 0 else (0.8 if ctx else 0.0)

                            citation = {
                                "text": ctx.text if ctx and hasattr(ctx, 'text') else "",
                                "file": ctx.title if ctx and hasattr(ctx, 'title') else "",
                                "relevance_score": relevance
                            }
                            citations.append(citation)
                            logger.info(f"Citation {i}: file='{citation['file']}', text_len={len(citation['text'])}, relevance={relevance:.3f}")

            # =============================================================
            # CACHE STORE: Save result for future similar queries
            # PERF: Fire-and-forget - don't block response on cache write
            # =============================================================
            if cache_enabled and org_id and answer and query_embedding is not None:
                async def cache_in_background():
                    """Background task to cache the result."""
                    try:
                        await semantic_cache_repository.cache_query(
                            org_id=org_id,
                            query_text=query,
                            query_embedding=query_embedding,
                            answer=answer,
                            citations=citations,
                            folder_filter=effective_folder_filter,
                            file_filter=effective_file_filter,
                            search_mode=search_mode,
                        )
                        logger.debug(f"Cached RAG response for future similar queries (file_filter={effective_file_filter})")
                    except Exception as cache_error:
                        logger.warning(f"Failed to cache RAG response: {cache_error}")

                # Fire-and-forget: don't await, let it complete in background
                asyncio.create_task(cache_in_background())

            result = {
                "success": True,
                "answer": answer,
                "citations": citations,
                "query": query,
                "search_mode": search_mode,
                "filters": {
                    "folder": effective_folder_filter,
                    "file": effective_file_filter
                },
                "cached": False,
                "processing_time_ms": elapsed_ms(start_time),
            }

            logger.info(f"RAG search completed: {len(citations)} citations in {result['processing_time_ms']:.1f}ms")
            return json.dumps(result)

        except Exception as e:
            logger.error(f"RAG search failed: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": str(e),
                "query": query
            })
