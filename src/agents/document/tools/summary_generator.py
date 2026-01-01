"""Summary generator tool using LLM."""

import json
import logging
import time
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import SummaryGeneratorInput, extract_llm_text, compute_content_hash
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class SummaryGeneratorTool(BaseTool):
    """Tool to generate document summary using LLM."""

    name: str = "summary_generator"
    description: str = """Generate a SUMMARY or OVERVIEW of document content.
    Use this tool ONLY when asked to summarize, create an overview, or condense the document.
    DO NOT use for questions, FAQs, or Q&A tasks.
    Takes document content and max_words. Returns a structured summary."""
    args_schema: Type[BaseModel] = SummaryGeneratorInput

    config: DocumentAgentConfig = Field(default_factory=DocumentAgentConfig)
    llm: Optional[Any] = None

    def _get_llm(self) -> Any:
        """Get or create LLM instance."""
        if self.llm is None:
            self.llm = init_chat_model(
                model=self.config.openai_model,
                model_provider="openai",
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key,
                use_responses_api=True,  # Required for gpt-5-nano
                timeout=300,  # 5 minutes for complex generation tasks
            )
        return self.llm

    def _run(
        self,
        content: str,
        document_name: str = "",
        max_words: int = 500,
        organization_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate summary from document content."""
        start_time = time.time()

        # Compute content hash for cache validation
        content_hash = compute_content_hash(content)

        # Check cache before generating
        try:
            from src.db.repositories.audit_repository import find_cached_generation
            from src.agents.core.audit_queue import enqueue_audit_event

            async def check_cache():
                return await find_cached_generation(
                    document_name=document_name,
                    generation_type='summary',
                    model=self.config.openai_model,
                    content_hash=content_hash,
                    organization_id=organization_id
                )

            cached = run_async(check_cache())
            if cached and cached.get('content', {}).get('summary'):
                cache_duration_ms = elapsed_ms(start_time)
                logger.info(f"Cache hit for summary: {document_name} ({cache_duration_ms:.1f}ms)")

                # Log cache hit via audit queue (non-blocking)
                enqueue_audit_event(
                    event_type='generation_cache_hit',
                    file_name=document_name,
                    organization_id=organization_id,
                    document_hash=content_hash,
                    details={
                        'generation_type': 'summary',
                        'generation_id': str(cached['id']),
                        'content_hash': content_hash
                    }
                )

                return json.dumps({
                    "success": True,
                    "summary": cached['content']['summary'],
                    "word_count": len(cached['content']['summary'].split()),
                    "processing_time_ms": cache_duration_ms,
                    "cached": True,
                    "content_hash": content_hash
                })

            # Log generation start via audit queue (non-blocking)
            enqueue_audit_event(
                event_type='generation_started',
                file_name=document_name,
                organization_id=organization_id,
                document_hash=content_hash,
                details={
                    'generation_type': 'summary',
                    'content_hash': content_hash,
                    'max_words': max_words
                }
            )

        except ImportError:
            logger.debug("Audit module not available, skipping cache check")
        except Exception as e:
            logger.warning(f"Cache check failed, proceeding with generation: {e}")

        prompt = f"""Generate a comprehensive summary of the following document.

Requirements:
- Maximum {max_words} words
- Start with a brief overview paragraph identifying the document type and purpose
- Include ALL key information: names, dates, amounts, quantities, addresses, reference numbers
- Organize content into logical sections if the document has distinct parts
- Preserve specific details that would be important for record-keeping or reference
- Use clear, professional language
- For invoices/receipts: include vendor, customer, items, prices, totals, payment terms
- For contracts: include parties, key terms, dates, obligations
- For reports: include main findings, data points, conclusions

Document content:
{content}

Comprehensive Summary:"""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            summary = extract_llm_text(response.content)

            duration_ms = elapsed_ms(start_time)
            word_count = len(summary.split())
            logger.info(f"Generated summary: {word_count} words in {duration_ms:.1f}ms")

            # Log generation complete via audit queue (non-blocking)
            try:
                from src.agents.core.audit_queue import enqueue_audit_event

                enqueue_audit_event(
                    event_type='generation_completed',
                    file_name=document_name,
                    organization_id=organization_id,
                    document_hash=content_hash,
                    details={
                        'generation_type': 'summary',
                        'content_hash': content_hash,
                        'word_count': word_count,
                        'processing_time_ms': duration_ms
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to log generation complete: {e}")

            return json.dumps({
                "success": True,
                "summary": summary,
                "word_count": word_count,
                "processing_time_ms": duration_ms,
                "cached": False,
                "content_hash": content_hash
            })

        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
