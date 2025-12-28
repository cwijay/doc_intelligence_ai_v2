"""Document Agent API endpoints.

Multi-tenancy: All endpoints are scoped by organization_id from request headers.
"""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_document_agent, get_org_id
from ..schemas.common import TokenUsage
from ..schemas.errors import DOCUMENT_ERROR_RESPONSES
from src.utils.timer_utils import elapsed_ms
from ..usage import (
    check_token_limit_before_processing,
    check_resource_limit_before_processing,
    log_resource_usage_async,
    check_quota,
    track_resource,
)
from ..schemas.documents import (
    DocumentProcessRequest,
    DocumentProcessResponse,
    SummarizeRequest,
    SummarizeResponse,
    FAQsRequest,
    FAQsResponse,
    QuestionsRequest,
    QuestionsResponse,
    GenerateAllRequest,
    GenerateAllResponse,
    GeneratedContent,
    FAQ,
    Question,
    RAGChatRequest,
    RAGChatResponse,
    RAGCitation,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/process",
    response_model=DocumentProcessResponse,
    responses=DOCUMENT_ERROR_RESPONSES,
    operation_id="processDocument",
    summary="Process document with flexible query",
)
async def process_document(
    request: DocumentProcessRequest,
    agent=Depends(get_document_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Process a document with the Document Agent using a flexible query.

    Supports flexible queries for generating summaries, FAQs, and comprehension questions.
    The query parameter allows natural language instructions for content generation.

    **Multi-tenancy**: Scoped by X-Organization-ID header.

    **Rate Limit**: 10 requests per 60 seconds per session.
    """
    start_time = time.time()

    # Check token limit before processing
    await check_token_limit_before_processing(org_id, estimated_tokens=2000)

    try:
        from src.agents.document.schemas import DocumentRequest, GenerationOptions

        # Build agent request
        options = None
        if request.options:
            options = GenerationOptions(
                num_faqs=request.options.num_faqs,
                num_questions=request.options.num_questions,
                summary_max_words=request.options.summary_max_words,
            )

        agent_request = DocumentRequest(
            document_name=request.document_name,
            parsed_file_path=request.parsed_file_path,
            query=request.query,
            session_id=request.session_id,
            user_id=request.user_id,
            options=options,
            organization_id=org_id,  # Multi-tenancy
        )

        # Process with agent
        response = await agent.process_request(agent_request)

        processing_time = elapsed_ms(start_time)

        # Token usage is now tracked via callback handlers in the agent
        # (see TokenTrackingCallbackHandler with use_context=True)

        return DocumentProcessResponse(
            success=response.success,
            message=response.message,
            document_name=request.document_name,
            content=GeneratedContent(
                summary=response.content.summary if response.content else None,
                faqs=[FAQ(question=f.question, answer=f.answer) for f in response.content.faqs] if response.content and response.content.faqs else None,
                questions=[Question(question=q.question, expected_answer=q.expected_answer, difficulty=q.difficulty) for q in response.content.questions] if response.content and response.content.questions else None,
            ) if response.content else None,
            token_usage=TokenUsage(
                prompt_tokens=response.token_usage.prompt_tokens,
                completion_tokens=response.token_usage.completion_tokens,
                total_tokens=response.token_usage.total_tokens,
                estimated_cost_usd=response.token_usage.estimated_cost_usd,
            ) if response.token_usage else None,
            processing_time_ms=processing_time,
            session_id=response.session_id,
        )

    except Exception as e:
        logger.exception(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses=DOCUMENT_ERROR_RESPONSES,
    operation_id="summarizeDocument",
    summary="Generate document summary",
)
async def summarize_document(
    request: SummarizeRequest,
    agent=Depends(get_document_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Generate a concise summary of the specified document.

    Uses Google Gemini to create a summary with configurable maximum word count.
    Results are cached in GCS for faster subsequent requests.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    start_time = time.time()

    try:
        # Check GCS cache first (unless force=true)
        if not request.force:
            from src.agents.document.gcs_cache import check_and_read_cached_summary

            cached = await check_and_read_cached_summary(
                parsed_file_path=request.parsed_file_path,
                document_name=request.document_name
            )
            if cached:
                processing_time = elapsed_ms(start_time)
                logger.info(f"GCS cache hit for summary: {request.document_name}")
                return SummarizeResponse(
                    success=True,
                    summary=cached.content,
                    word_count=cached.word_count,
                    cached=True,
                    processing_time_ms=processing_time,
                )

        # Not cached or force=true, generate via agent
        result = await agent.generate_summary(
            document_name=request.document_name,
            parsed_file_path=request.parsed_file_path,
            max_words=request.max_words,
            organization_id=org_id,
        )

        processing_time = elapsed_ms(start_time)

        if result:
            return SummarizeResponse(
                success=True,
                summary=result,
                word_count=len(result.split()),
                cached=False,
                processing_time_ms=processing_time,
            )
        else:
            return SummarizeResponse(
                success=False,
                error="Failed to generate summary",
                processing_time_ms=processing_time,
            )

    except Exception as e:
        logger.exception(f"Summary generation failed: {e}")
        return SummarizeResponse(
            success=False,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time),
        )


@router.post(
    "/faqs",
    response_model=FAQsResponse,
    responses=DOCUMENT_ERROR_RESPONSES,
    operation_id="generateFaqs",
    summary="Generate FAQs from document",
)
async def generate_faqs(
    request: FAQsRequest,
    agent=Depends(get_document_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Generate Frequently Asked Questions (FAQs) from the specified document.

    Uses Google Gemini to extract key questions and answers from document content.
    Results are cached in GCS for faster subsequent requests.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    start_time = time.time()

    try:
        # Check GCS cache first (unless force=true)
        if not request.force:
            from src.agents.document.gcs_cache import check_and_read_cached_faqs

            cached = await check_and_read_cached_faqs(
                parsed_file_path=request.parsed_file_path,
                document_name=request.document_name
            )
            if cached:
                processing_time = elapsed_ms(start_time)
                logger.info(f"GCS cache hit for FAQs: {request.document_name}")
                faqs = [FAQ(question=f.question, answer=f.answer) for f in cached.faqs]
                return FAQsResponse(
                    success=True,
                    faqs=faqs,
                    count=cached.count,
                    cached=True,
                    processing_time_ms=processing_time,
                )

        # Not cached or force=true, generate via agent
        result = await agent.generate_faqs(
            document_name=request.document_name,
            parsed_file_path=request.parsed_file_path,
            num_faqs=request.num_faqs,
            organization_id=org_id,
        )

        processing_time = elapsed_ms(start_time)

        if result:
            faqs = [FAQ(question=f.question, answer=f.answer) for f in result]
            return FAQsResponse(
                success=True,
                faqs=faqs,
                count=len(faqs),
                cached=False,
                processing_time_ms=processing_time,
            )
        else:
            return FAQsResponse(
                success=False,
                error="Failed to generate FAQs",
                processing_time_ms=processing_time,
            )

    except Exception as e:
        logger.exception(f"FAQ generation failed: {e}")
        return FAQsResponse(
            success=False,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time),
        )


@router.post(
    "/questions",
    response_model=QuestionsResponse,
    responses=DOCUMENT_ERROR_RESPONSES,
    operation_id="generateQuestions",
    summary="Generate comprehension questions",
)
async def generate_questions(
    request: QuestionsRequest,
    agent=Depends(get_document_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Generate comprehension questions for the specified document.

    Uses Google Gemini to create questions with varying difficulty levels (easy, medium, hard).
    Includes expected answers for each question. Results are cached in GCS.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    start_time = time.time()

    try:
        # Check GCS cache first (unless force=true)
        if not request.force:
            from src.agents.document.gcs_cache import check_and_read_cached_questions

            cached = await check_and_read_cached_questions(
                parsed_file_path=request.parsed_file_path,
                document_name=request.document_name
            )
            if cached:
                processing_time = elapsed_ms(start_time)
                logger.info(f"GCS cache hit for questions: {request.document_name}")
                questions = [
                    Question(
                        question=q.question,
                        expected_answer=q.expected_answer,
                        difficulty=q.difficulty,
                    )
                    for q in cached.questions
                ]
                return QuestionsResponse(
                    success=True,
                    questions=questions,
                    count=cached.count,
                    difficulty_distribution=cached.difficulty_distribution,
                    cached=True,
                    processing_time_ms=processing_time,
                )

        # Not cached or force=true, generate via agent
        result = await agent.generate_questions(
            document_name=request.document_name,
            parsed_file_path=request.parsed_file_path,
            num_questions=request.num_questions,
            organization_id=org_id,
        )

        processing_time = elapsed_ms(start_time)

        if result:
            questions = [
                Question(
                    question=q.question,
                    expected_answer=q.expected_answer,
                    difficulty=q.difficulty,
                )
                for q in result
            ]

            # Calculate difficulty distribution
            distribution = {"easy": 0, "medium": 0, "hard": 0}
            for q in questions:
                if q.difficulty.lower() in distribution:
                    distribution[q.difficulty.lower()] += 1

            return QuestionsResponse(
                success=True,
                questions=questions,
                count=len(questions),
                difficulty_distribution=distribution,
                cached=False,
                processing_time_ms=processing_time,
            )
        else:
            return QuestionsResponse(
                success=False,
                error="Failed to generate questions",
                processing_time_ms=processing_time,
            )

    except Exception as e:
        logger.exception(f"Question generation failed: {e}")
        return QuestionsResponse(
            success=False,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time),
        )


@router.post(
    "/generate-all",
    response_model=GenerateAllResponse,
    responses=DOCUMENT_ERROR_RESPONSES,
    operation_id="generateAllContent",
    summary="Generate all content types",
)
async def generate_all_content(
    request: GenerateAllRequest,
    agent=Depends(get_document_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Generate all content types (summary, FAQs, questions) for a document in a single request.

    Combines summary generation, FAQ extraction, and question generation into one operation.
    Results are cached in GCS for faster subsequent requests.

    **Multi-tenancy**: Scoped by X-Organization-ID header.
    """
    start_time = time.time()

    try:
        # Check GCS cache first (unless force=true)
        if not request.force:
            from src.agents.document.gcs_cache import (
                check_and_read_cached_summary,
                check_and_read_cached_faqs,
                check_and_read_cached_questions
            )

            # Check all three caches
            cached_summary = await check_and_read_cached_summary(
                parsed_file_path=request.parsed_file_path,
                document_name=request.document_name
            )
            cached_faqs = await check_and_read_cached_faqs(
                parsed_file_path=request.parsed_file_path,
                document_name=request.document_name
            )
            cached_questions = await check_and_read_cached_questions(
                parsed_file_path=request.parsed_file_path,
                document_name=request.document_name
            )

            # If ALL content is cached, return from cache
            if cached_summary and cached_faqs and cached_questions:
                processing_time = elapsed_ms(start_time)
                logger.info(f"GCS cache hit for all content: {request.document_name}")
                return GenerateAllResponse(
                    success=True,
                    document_name=request.document_name,
                    summary=cached_summary.content,
                    faqs=[FAQ(question=f.question, answer=f.answer) for f in cached_faqs.faqs],
                    questions=[Question(question=q.question, expected_answer=q.expected_answer, difficulty=q.difficulty) for q in cached_questions.questions],
                    cached=True,
                    processing_time_ms=processing_time,
                )

        # Not all cached or force=true, generate via agent
        # Build options
        options = None
        if request.options:
            from src.agents.document.schemas import GenerationOptions
            options = GenerationOptions(
                num_faqs=request.options.num_faqs,
                num_questions=request.options.num_questions,
                summary_max_words=request.options.summary_max_words,
            )

        result = await agent.generate_all(
            document_name=request.document_name,
            parsed_file_path=request.parsed_file_path,
            options=options,
            organization_id=org_id,
        )

        processing_time = elapsed_ms(start_time)

        if result:
            return GenerateAllResponse(
                success=True,
                document_name=request.document_name,
                summary=result.summary,
                faqs=[FAQ(question=f.question, answer=f.answer) for f in result.faqs] if result.faqs else None,
                questions=[Question(question=q.question, expected_answer=q.expected_answer, difficulty=q.difficulty) for q in result.questions] if result.questions else None,
                cached=False,
                processing_time_ms=processing_time,
            )
        else:
            return GenerateAllResponse(
                success=False,
                document_name=request.document_name,
                error="Failed to generate content",
                processing_time_ms=processing_time,
            )

    except Exception as e:
        logger.exception(f"Generate all failed: {e}")
        return GenerateAllResponse(
            success=False,
            document_name=request.document_name,
            error=str(e),
            processing_time_ms=elapsed_ms(start_time),
        )


@router.post(
    "/chat",
    response_model=RAGChatResponse,
    responses=DOCUMENT_ERROR_RESPONSES,
    operation_id="chatWithDocuments",
    summary="Conversational RAG chat with documents",
)
async def chat_with_documents(
    request: RAGChatRequest,
    agent=Depends(get_document_agent),
    org_id: str = Depends(get_org_id),
):
    """
    Conversational RAG - chat with documents using natural language.

    Supports:
    - Semantic, keyword, or hybrid search modes
    - Folder and file filtering
    - Conversation continuity via session_id
    - Answer generation with citations

    **Search Scopes**:
    - Single File: Set `file_filter` to target file name
    - Folder: Set `folder_filter` to target folder name
    - Org-wide: Leave both filters empty

    **Multi-tenancy**: Scoped by X-Organization-ID header.

    **Usage Tracking**: Each search query counts against the monthly file_search_queries limit.
    """
    start_time = time.time()

    # Check file search query limit before processing
    await check_resource_limit_before_processing(
        org_id, resource_type="file_search_queries", estimated_usage=1
    )

    try:
        # Use organization_name from request (required field)
        organization_name = request.organization_name

        # Use the agent's chat method
        response = await agent.chat(
            query=request.query,
            organization_name=organization_name,
            session_id=request.session_id,
            folder_filter=request.folder_filter,
            file_filter=request.file_filter,
            search_mode=request.search_mode,
            organization_id=org_id,
        )

        processing_time = elapsed_ms(start_time)

        # Extract citations from response if available
        citations = []
        if hasattr(response, 'citations') and response.citations:
            citations = [
                RAGCitation(
                    text=c.get('text', ''),
                    file=c.get('file', ''),
                    relevance_score=c.get('relevance_score', 0.0),
                    folder_name=c.get('folder_name'),
                )
                for c in response.citations
            ]

        # Log file search query usage (non-blocking)
        log_resource_usage_async(
            org_id=org_id,
            resource_type="file_search_queries",
            amount=1,
            user_id=None,
            extra_data={
                "query": request.query[:100] if request.query else None,
                "search_mode": request.search_mode,
                "folder_filter": request.folder_filter,
                "file_filter": request.file_filter,
                "session_id": request.session_id,
            },
        )

        return RAGChatResponse(
            success=response.success,
            answer=response.response_text or response.message,  # Use agent's response, fallback to message
            citations=citations,
            query=request.query,
            search_mode=request.search_mode,
            filters={
                "folder": request.folder_filter,
                "file": request.file_filter,
            },
            session_id=response.session_id,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        logger.exception(f"RAG chat failed: {e}")
        processing_time = elapsed_ms(start_time)
        return RAGChatResponse(
            success=False,
            answer="",
            citations=[],
            query=request.query,
            search_mode=request.search_mode,
            filters={
                "folder": request.folder_filter,
                "file": request.file_filter,
            },
            session_id=request.session_id or "",
            processing_time_ms=processing_time,
            error=str(e),
        )
