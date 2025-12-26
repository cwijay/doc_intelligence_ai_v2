"""Summary generator tool using LLM."""

import json
import logging
import time
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import SummaryGeneratorInput, extract_llm_text, compute_content_hash
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class SummaryGeneratorTool(BaseTool):
    """Tool to generate document summary using LLM."""

    name: str = "summary_generator"
    description: str = """Generate a concise summary of document content.
    Takes document content and max_words as input.
    Returns a well-structured summary capturing key points."""
    args_schema: Type[BaseModel] = SummaryGeneratorInput

    config: DocumentAgentConfig = Field(default_factory=DocumentAgentConfig)
    llm: Optional[ChatGoogleGenerativeAI] = None

    def _get_llm(self) -> ChatGoogleGenerativeAI:
        """Get or create LLM instance."""
        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.gemini_model,
                google_api_key=self.config.google_api_key,
                temperature=self.config.temperature
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
            from src.db.repositories.audit_repository import find_cached_generation, log_event

            async def check_cache():
                cached = await find_cached_generation(
                    document_name=document_name,
                    generation_type='summary',
                    model=self.config.gemini_model,
                    content_hash=content_hash,
                    organization_id=organization_id
                )
                if cached:
                    await log_event(
                        event_type='generation_cache_hit',
                        file_name=document_name,
                        organization_id=organization_id,
                        details={
                            'generation_type': 'summary',
                            'generation_id': str(cached['id']),
                            'content_hash': content_hash
                        }
                    )
                return cached

            cached = run_async(check_cache())
            if cached and cached.get('content', {}).get('summary'):
                cache_duration_ms = elapsed_ms(start_time)
                logger.info(f"Cache hit for summary: {document_name} ({cache_duration_ms:.1f}ms)")
                return json.dumps({
                    "success": True,
                    "summary": cached['content']['summary'],
                    "word_count": len(cached['content']['summary'].split()),
                    "processing_time_ms": cache_duration_ms,
                    "cached": True,
                    "content_hash": content_hash
                })

            # Log generation start
            async def log_start():
                await log_event(
                    event_type='generation_started',
                    file_name=document_name,
                    organization_id=organization_id,
                    details={
                        'generation_type': 'summary',
                        'content_hash': content_hash,
                        'max_words': max_words
                    }
                )
            run_async(log_start())

        except ImportError:
            logger.debug("Audit module not available, skipping cache check")
        except Exception as e:
            logger.warning(f"Cache check failed, proceeding with generation: {e}")

        prompt = f"""Generate a concise summary of the following document.
The summary should:
- Be no more than {max_words} words
- Capture the main topics and key points
- Be well-structured and easy to read
- Use clear, professional language

Document content:
{content}

Summary:"""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            summary = extract_llm_text(response.content)

            duration_ms = elapsed_ms(start_time)
            word_count = len(summary.split())
            logger.info(f"Generated summary: {word_count} words in {duration_ms:.1f}ms")

            # Log generation complete
            try:
                from src.db.repositories.audit_repository import log_event

                async def log_complete():
                    await log_event(
                        event_type='generation_completed',
                        file_name=document_name,
                        organization_id=organization_id,
                        details={
                            'generation_type': 'summary',
                            'content_hash': content_hash,
                            'word_count': word_count,
                            'processing_time_ms': duration_ms
                        }
                    )
                run_async(log_complete())
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
