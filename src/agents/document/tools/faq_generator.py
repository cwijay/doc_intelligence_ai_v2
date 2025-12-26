"""FAQ generator tool using LLM."""

import json
import logging
import time
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import FAQGeneratorInput, extract_llm_text, compute_content_hash
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class FAQGeneratorTool(BaseTool):
    """Tool to generate FAQs from document content using LLM."""

    name: str = "faq_generator"
    description: str = """Generate frequently asked questions and answers from document content.
    Takes document content and number of FAQs to generate.
    Returns a list of question-answer pairs."""
    args_schema: Type[BaseModel] = FAQGeneratorInput

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
        num_faqs: int = 5,
        organization_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate FAQs from document content."""
        start_time = time.time()

        # Compute content hash for cache validation
        content_hash = compute_content_hash(content)

        # Check cache before generating
        try:
            from src.db.repositories.audit_repository import find_cached_generation, log_event

            async def check_cache():
                cached = await find_cached_generation(
                    document_name=document_name,
                    generation_type='faqs',
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
                            'generation_type': 'faqs',
                            'generation_id': str(cached['id']),
                            'content_hash': content_hash
                        }
                    )
                return cached

            cached = run_async(check_cache())
            if cached and cached.get('content', {}).get('faqs'):
                cache_duration_ms = elapsed_ms(start_time)
                logger.info(f"Cache hit for FAQs: {document_name} ({cache_duration_ms:.1f}ms)")
                return json.dumps({
                    "success": True,
                    "faqs": cached['content']['faqs'],
                    "count": len(cached['content']['faqs']),
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
                        'generation_type': 'faqs',
                        'content_hash': content_hash,
                        'num_faqs': num_faqs
                    }
                )
            run_async(log_start())

        except ImportError:
            logger.debug("Audit module not available, skipping cache check")
        except Exception as e:
            logger.warning(f"Cache check failed, proceeding with generation: {e}")

        prompt = f"""Based on the following document, generate exactly {num_faqs} frequently asked questions and their answers.

Requirements:
- Questions should be what a reader would commonly ask about this content
- Answers should be accurate and based only on the document content
- Answers should be concise but complete
- Cover different aspects of the document

Return the FAQs in this exact JSON format:
{{
  "faqs": [
    {{"question": "...", "answer": "..."}},
    {{"question": "...", "answer": "..."}}
  ]
}}

Document content:
{content}

JSON Response:"""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            response_text = extract_llm_text(response.content)

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_text)
            faqs = parsed.get("faqs", [])

            duration_ms = elapsed_ms(start_time)
            logger.info(f"Generated {len(faqs)} FAQs in {duration_ms:.1f}ms")

            # Log generation complete
            try:
                from src.db.repositories.audit_repository import log_event

                async def log_complete():
                    await log_event(
                        event_type='generation_completed',
                        file_name=document_name,
                        organization_id=organization_id,
                        details={
                            'generation_type': 'faqs',
                            'content_hash': content_hash,
                            'count': len(faqs),
                            'processing_time_ms': duration_ms
                        }
                    )
                run_async(log_complete())
            except Exception as e:
                logger.debug(f"Failed to log generation complete: {e}")

            return json.dumps({
                "success": True,
                "faqs": faqs,
                "count": len(faqs),
                "processing_time_ms": duration_ms,
                "cached": False,
                "content_hash": content_hash
            })

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse FAQ response: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to parse LLM response as JSON: {e}"
            })
        except Exception as e:
            logger.error(f"FAQ generation failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
