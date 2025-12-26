"""Question generator tool using LLM."""

import json
import logging
import time
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import QuestionGeneratorInput, extract_llm_text, compute_content_hash
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class QuestionGeneratorTool(BaseTool):
    """Tool to generate comprehension questions from document content using LLM."""

    name: str = "question_generator"
    description: str = """Generate comprehension questions from document content.
    Takes document content and number of questions to generate.
    Returns questions with expected answers and difficulty levels."""
    args_schema: Type[BaseModel] = QuestionGeneratorInput

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
        num_questions: int = 10,
        organization_id: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate comprehension questions from document content."""
        start_time = time.time()

        # Compute content hash for cache validation
        content_hash = compute_content_hash(content)

        # Check cache before generating
        try:
            from src.db.repositories.audit_repository import find_cached_generation, log_event

            async def check_cache():
                cached = await find_cached_generation(
                    document_name=document_name,
                    generation_type='questions',
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
                            'generation_type': 'questions',
                            'generation_id': str(cached['id']),
                            'content_hash': content_hash
                        }
                    )
                return cached

            cached = run_async(check_cache())
            if cached and cached.get('content', {}).get('questions'):
                cache_duration_ms = elapsed_ms(start_time)
                cached_questions = cached['content']['questions']
                logger.info(f"Cache hit for questions: {document_name} ({cache_duration_ms:.1f}ms)")

                # Count by difficulty
                difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
                for q in cached_questions:
                    d = q.get("difficulty", "medium").lower()
                    if d in difficulty_counts:
                        difficulty_counts[d] += 1

                return json.dumps({
                    "success": True,
                    "questions": cached_questions,
                    "count": len(cached_questions),
                    "difficulty_distribution": difficulty_counts,
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
                        'generation_type': 'questions',
                        'content_hash': content_hash,
                        'num_questions': num_questions
                    }
                )
            run_async(log_start())

        except ImportError:
            logger.debug("Audit module not available, skipping cache check")
        except Exception as e:
            logger.warning(f"Cache check failed, proceeding with generation: {e}")

        # Calculate distribution of difficulty levels
        easy_count = num_questions // 3
        hard_count = num_questions // 3
        medium_count = num_questions - easy_count - hard_count

        prompt = f"""Based on the following document, generate exactly {num_questions} comprehension questions.

Requirements:
- Generate approximately {easy_count} easy, {medium_count} medium, and {hard_count} hard questions
- Easy questions: basic recall and understanding
- Medium questions: application and analysis
- Hard questions: synthesis and evaluation
- Each question should have an expected answer based on the document
- Questions should cover different parts of the document

Return the questions in this exact JSON format:
{{
  "questions": [
    {{"question": "...", "expected_answer": "...", "difficulty": "easy"}},
    {{"question": "...", "expected_answer": "...", "difficulty": "medium"}},
    {{"question": "...", "expected_answer": "...", "difficulty": "hard"}}
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
            questions = parsed.get("questions", [])

            duration_ms = elapsed_ms(start_time)
            logger.info(f"Generated {len(questions)} questions in {duration_ms:.1f}ms")

            # Count by difficulty
            difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
            for q in questions:
                d = q.get("difficulty", "medium").lower()
                if d in difficulty_counts:
                    difficulty_counts[d] += 1

            # Log generation complete
            try:
                from src.db.repositories.audit_repository import log_event

                async def log_complete():
                    await log_event(
                        event_type='generation_completed',
                        file_name=document_name,
                        organization_id=organization_id,
                        details={
                            'generation_type': 'questions',
                            'content_hash': content_hash,
                            'count': len(questions),
                            'difficulty_distribution': difficulty_counts,
                            'processing_time_ms': duration_ms
                        }
                    )
                run_async(log_complete())
            except Exception as e:
                logger.debug(f"Failed to log generation complete: {e}")

            return json.dumps({
                "success": True,
                "questions": questions,
                "count": len(questions),
                "difficulty_distribution": difficulty_counts,
                "processing_time_ms": duration_ms,
                "cached": False,
                "content_hash": content_hash
            })

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse question response: {e}")
            return json.dumps({
                "success": False,
                "error": f"Failed to parse LLM response as JSON: {e}"
            })
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            return json.dumps({
                "success": False,
                "error": str(e)
            })
