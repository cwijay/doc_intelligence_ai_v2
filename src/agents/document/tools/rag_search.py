"""RAG search tool using Gemini File Search with semantic caching."""

import json
import logging
import time
from typing import Optional, Type

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

        try:
            from src.rag.gemini_file_store import query_store, generate_store_display_name, client as gemini_client
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
                            folder_filter=folder_filter,
                            file_filter=file_filter,
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
                                    "folder": folder_filter,
                                    "file": file_filter
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

                store = gemini_client.file_search_stores.get(name=gemini_store_id)
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
                    file_name_filter=file_filter,
                    folder_name_filter=folder_filter,
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
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'grounding_metadata'):
                        grounding = candidate.grounding_metadata
                        if hasattr(grounding, 'grounding_chunks'):
                            for chunk in grounding.grounding_chunks:
                                citation = {
                                    "text": chunk.text if hasattr(chunk, 'text') else "",
                                    "file": chunk.source if hasattr(chunk, 'source') else "",
                                    "relevance_score": chunk.relevance_score if hasattr(chunk, 'relevance_score') else 0.0
                                }
                                citations.append(citation)

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
                            folder_filter=folder_filter,
                            file_filter=file_filter,
                            search_mode=search_mode,
                        )
                        logger.debug(f"Cached RAG response for future similar queries")
                    except Exception as cache_error:
                        logger.warning(f"Failed to cache RAG response: {cache_error}")

                return {
                    "success": True,
                    "answer": answer,
                    "citations": citations,
                    "query": query,
                    "search_mode": search_mode,
                    "filters": {
                        "folder": folder_filter,
                        "file": file_filter
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

        try:
            from src.rag.gemini_file_store import query_store, generate_store_display_name, client as gemini_client
            from src.db.repositories import rag_repository
            from src.db.repositories import semantic_cache_repository
            from src.rag.embeddings import get_query_embedding_sync

            # Generate normalized store name for DB lookup
            store_name = generate_store_display_name(organization_name)

            # Look up store from PostgreSQL (faster than Gemini API listing)
            store_record = await rag_repository.get_store_by_display_name(store_name)
            if not store_record:
                return json.dumps({
                    "success": False,
                    "error": f"File search store not found for organization: {organization_name}. Please ensure documents have been ingested first.",
                    "query": query
                })

            org_id = store_record.get("organization_id")
            query_embedding = None

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
                        folder_filter=folder_filter,
                        file_filter=file_filter,
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
                                "folder": folder_filter,
                                "file": file_filter
                            },
                            "cached": True,
                            "cache_similarity": cached["similarity"],
                            "processing_time_ms": elapsed_ms(start_time),
                        }
                        logger.info(f"RAG search completed (CACHED): {len(result.get('citations', []))} citations in {result['processing_time_ms']:.1f}ms")
                        return json.dumps(result)
                except Exception as cache_error:
                    logger.warning(f"Semantic cache lookup failed: {cache_error}")
                    query_embedding = None

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

            store = gemini_client.file_search_stores.get(name=gemini_store_id)
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
                file_name_filter=file_filter,
                folder_name_filter=folder_filter,
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
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    grounding = candidate.grounding_metadata
                    if hasattr(grounding, 'grounding_chunks'):
                        for chunk in grounding.grounding_chunks:
                            citation = {
                                "text": chunk.text if hasattr(chunk, 'text') else "",
                                "file": chunk.source if hasattr(chunk, 'source') else "",
                                "relevance_score": chunk.relevance_score if hasattr(chunk, 'relevance_score') else 0.0
                            }
                            citations.append(citation)

            # =============================================================
            # CACHE STORE: Save result for future similar queries
            # =============================================================
            if semantic_cache_repository.is_cache_enabled() and org_id and answer:
                try:
                    # Generate embedding if we didn't already
                    if query_embedding is None:
                        query_embedding = get_query_embedding_sync(query)

                    await semantic_cache_repository.cache_query(
                        org_id=org_id,
                        query_text=query,
                        query_embedding=query_embedding,
                        answer=answer,
                        citations=citations,
                        folder_filter=folder_filter,
                        file_filter=file_filter,
                        search_mode=search_mode,
                    )
                    logger.debug(f"Cached RAG response for future similar queries")
                except Exception as cache_error:
                    logger.warning(f"Failed to cache RAG response: {cache_error}")

            result = {
                "success": True,
                "answer": answer,
                "citations": citations,
                "query": query,
                "search_mode": search_mode,
                "filters": {
                    "folder": folder_filter,
                    "file": file_filter
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
