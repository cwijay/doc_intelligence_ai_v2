"""Generation helpers for bulk document processing.

Provides helper functions for content generation in the state graph.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of content generation."""
    summary: Optional[str] = None
    faqs: Optional[List[dict]] = None
    questions: Optional[List[dict]] = None
    errors: List[str] = None
    token_usage: int = 0

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class GenerationContext:
    """Context for generation operations."""
    content: str
    base_name: str
    org_id: str
    bulk_job_id: str
    document_id: str
    parsed_path: str


def create_generation_tools():
    """Create document generation tool instances.

    Returns:
        Tuple of (summary_tool, faq_tool, question_tool, persist_tool, config)
    """
    from src.agents.document.config import DocumentAgentConfig
    from src.agents.document.tools import (
        SummaryGeneratorTool,
        FAQGeneratorTool,
        QuestionGeneratorTool,
        ContentPersistTool,
    )

    config = DocumentAgentConfig()
    return (
        SummaryGeneratorTool(config=config),
        FAQGeneratorTool(config=config),
        QuestionGeneratorTool(config=config),
        ContentPersistTool(config=config),
        config,
    )


async def run_generators_parallel(
    content: str,
    base_name: str,
    org_id: str,
    generate_summary: bool,
    generate_faqs: bool,
    generate_questions: bool,
    summary_max_words: int,
    num_faqs: int,
    num_questions: int,
    timeout_seconds: float,
) -> Tuple[Any, Any, Any]:
    """Run content generators in parallel.

    Args:
        content: Document content to process
        base_name: Document base name
        org_id: Organization ID
        generate_summary: Whether to generate summary
        generate_faqs: Whether to generate FAQs
        generate_questions: Whether to generate questions
        summary_max_words: Max words for summary
        num_faqs: Number of FAQs to generate
        num_questions: Number of questions to generate
        timeout_seconds: Timeout for generation

    Returns:
        Tuple of (summary_result, faqs_result, questions_result)
    """
    summary_tool, faq_tool, question_tool, _, _ = create_generation_tools()
    loop = asyncio.get_running_loop()

    async def run_summary():
        if not generate_summary:
            return None
        return await loop.run_in_executor(
            None,
            lambda: summary_tool._run(
                content=content,
                document_name=base_name,
                max_words=summary_max_words,
                organization_id=org_id,
            )
        )

    async def run_faqs():
        if not generate_faqs:
            return None
        return await loop.run_in_executor(
            None,
            lambda: faq_tool._run(
                content=content,
                document_name=base_name,
                num_faqs=num_faqs,
                organization_id=org_id,
            )
        )

    async def run_questions():
        if not generate_questions:
            return None
        return await loop.run_in_executor(
            None,
            lambda: question_tool._run(
                content=content,
                document_name=base_name,
                num_questions=num_questions,
                organization_id=org_id,
            )
        )

    return await asyncio.wait_for(
        asyncio.gather(
            run_summary(),
            run_faqs(),
            run_questions(),
            return_exceptions=True,
        ),
        timeout=timeout_seconds,
    )


def parse_generation_results(
    summary_result: Any,
    faqs_result: Any,
    questions_result: Any,
) -> GenerationResult:
    """Parse JSON results from generation tools.

    Args:
        summary_result: Raw summary tool result
        faqs_result: Raw FAQs tool result
        questions_result: Raw questions tool result

    Returns:
        GenerationResult with parsed content and any errors
    """
    result = GenerationResult()

    # Parse summary
    if summary_result:
        if isinstance(summary_result, Exception):
            result.errors.append(f"Summary: {summary_result}")
        else:
            try:
                parsed = json.loads(summary_result)
                if parsed.get("success"):
                    result.summary = parsed.get("summary")
                else:
                    result.errors.append(f"Summary: {parsed.get('error')}")
            except json.JSONDecodeError as e:
                result.errors.append(f"Summary parse error: {e}")

    # Parse FAQs
    if faqs_result:
        if isinstance(faqs_result, Exception):
            result.errors.append(f"FAQs: {faqs_result}")
        else:
            try:
                parsed = json.loads(faqs_result)
                if parsed.get("success"):
                    result.faqs = parsed.get("faqs")
                else:
                    result.errors.append(f"FAQs: {parsed.get('error')}")
            except json.JSONDecodeError as e:
                result.errors.append(f"FAQs parse error: {e}")

    # Parse questions
    if questions_result:
        if isinstance(questions_result, Exception):
            result.errors.append(f"Questions: {questions_result}")
        else:
            try:
                parsed = json.loads(questions_result)
                if parsed.get("success"):
                    result.questions = parsed.get("questions")
                else:
                    result.errors.append(f"Questions: {parsed.get('error')}")
            except json.JSONDecodeError as e:
                result.errors.append(f"Questions parse error: {e}")

    return result


async def persist_generated_content(
    result: GenerationResult,
    context: GenerationContext,
) -> None:
    """Persist generated content to GCS.

    Args:
        result: Generation result with content
        context: Generation context with paths
    """
    if not (result.summary or result.faqs or result.questions):
        return

    _, _, _, persist_tool, _ = create_generation_tools()
    loop = asyncio.get_running_loop()

    persist_input = {
        "document_name": context.base_name,
        "parsed_file_path": context.parsed_path,
        "organization_id": context.org_id,
    }

    if result.summary:
        persist_input["summary"] = result.summary
    if result.faqs:
        persist_input["faqs"] = json.dumps(result.faqs)
    if result.questions:
        persist_input["questions"] = json.dumps(result.questions)

    await loop.run_in_executor(
        None,
        lambda: persist_tool._run(**persist_input)
    )


def estimate_token_usage(result: GenerationResult) -> int:
    """Estimate token usage from generation result.

    Args:
        result: Generation result

    Returns:
        Estimated token count
    """
    token_usage = 0
    if result.summary:
        token_usage += len(result.summary.split()) * 2
    if result.faqs:
        token_usage += len(result.faqs) * 100
    if result.questions:
        token_usage += len(result.questions) * 100
    return token_usage


def track_generation_usage(
    org_id: str,
    context: GenerationContext,
    result: GenerationResult,
    model: str,
) -> None:
    """Track token usage for generation (non-blocking).

    Args:
        org_id: Organization ID
        context: Generation context
        result: Generation result
        model: Model used for generation
    """
    try:
        from src.core.usage import enqueue_token_usage
    except ImportError:
        return

    token_usage = estimate_token_usage(result)
    if token_usage <= 0:
        return

    try:
        enqueue_token_usage(
            org_id=org_id,
            feature="bulk_processing",
            provider="openai",
            model=model,
            input_tokens=token_usage // 2,
            output_tokens=token_usage // 2,
            metadata={
                "bulk_job_id": context.bulk_job_id,
                "document_id": context.document_id,
                "summary_generated": bool(result.summary),
                "faqs_count": len(result.faqs) if result.faqs else 0,
                "questions_count": len(result.questions) if result.questions else 0,
            },
        )
    except Exception as e:
        logger.warning(f"Failed to enqueue token usage: {e}")


async def load_document_content(parsed_path: str, cached_content: Optional[str] = None) -> str:
    """Load document content from cache or GCS.

    Args:
        parsed_path: Path to parsed document
        cached_content: Pre-loaded content if available

    Returns:
        Document content
    """
    if cached_content:
        return cached_content

    from src.storage import get_storage
    storage = get_storage()
    return await storage.read(parsed_path, use_prefix=False)
