"""Question generator tool using LLM with parallel generation."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import QuestionGeneratorInput, extract_llm_text, compute_content_hash
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)

# Difficulty descriptions for parallel generation
DIFFICULTY_DESCRIPTIONS = {
    "easy": "basic recall and understanding (who, what, where, when)",
    "medium": "application and analysis (how, why, compare)",
    "hard": "synthesis and evaluation (analyze, evaluate, propose)",
}


class QuestionGeneratorTool(BaseTool):
    """Tool to generate comprehension questions from document content using LLM."""

    name: str = "question_generator"
    description: str = """Generate COMPREHENSION QUESTIONS with answers from document content.
    Use this tool ONLY when asked to create questions, quiz questions, test questions, or comprehension exercises.
    DO NOT use for summaries or FAQs.
    Returns questions with expected answers and difficulty levels (easy/medium/hard)."""
    args_schema: Type[BaseModel] = QuestionGeneratorInput

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
                timeout=120,  # 2 minutes per batch (parallel calls)
            )
        return self.llm

    def _generate_questions_for_difficulty(
        self,
        content: str,
        difficulty: str,
        count: int,
    ) -> List[Dict]:
        """Generate questions for a single difficulty level.

        Args:
            content: Document content
            difficulty: Difficulty level (easy, medium, hard)
            count: Number of questions to generate

        Returns:
            List of question dictionaries
        """
        description = DIFFICULTY_DESCRIPTIONS.get(difficulty, "general understanding")

        prompt = f"""Generate exactly {count} comprehension questions at the "{difficulty}" difficulty level.

Difficulty "{difficulty}": {description}

Requirements:
- Each question should have a clear expected answer based on the document
- Questions should test {description}
- Keep expected answers concise (1-3 sentences)

Return ONLY a JSON array:
[
  {{"question": "...", "expected_answer": "...", "difficulty": "{difficulty}"}},
  {{"question": "...", "expected_answer": "...", "difficulty": "{difficulty}"}}
]

Document content:
{content}

JSON Array:"""

        try:
            llm = self._get_llm()
            response = llm.invoke(prompt)
            response_text = extract_llm_text(response.content)

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Handle both array and object responses
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                questions = parsed
            elif isinstance(parsed, dict) and "questions" in parsed:
                questions = parsed["questions"]
            else:
                questions = []

            # Ensure all questions have correct difficulty
            for q in questions:
                q["difficulty"] = difficulty

            logger.debug(f"Generated {len(questions)} {difficulty} questions")
            return questions

        except Exception as e:
            logger.warning(f"Failed to generate {difficulty} questions: {e}")
            return []

    def _generate_parallel(
        self,
        content: str,
        num_questions: int,
    ) -> List[Dict]:
        """Generate questions in parallel by difficulty level.

        Splits work into 3 parallel LLM calls for easy/medium/hard questions.
        """
        # Calculate distribution
        easy_count = num_questions // 3
        hard_count = num_questions // 3
        medium_count = num_questions - easy_count - hard_count

        tasks = [
            ("easy", easy_count),
            ("medium", medium_count),
            ("hard", hard_count),
        ]

        all_questions = []
        start_time = time.time()

        # Run parallel generation
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._generate_questions_for_difficulty,
                    content,
                    difficulty,
                    count,
                ): difficulty
                for difficulty, count in tasks
                if count > 0
            }

            for future in as_completed(futures):
                difficulty = futures[future]
                try:
                    questions = future.result()
                    all_questions.extend(questions)
                except Exception as e:
                    logger.error(f"Parallel generation failed for {difficulty}: {e}")

        parallel_time = (time.time() - start_time) * 1000
        logger.info(
            f"Parallel generation: {len(all_questions)} questions in {parallel_time:.1f}ms"
        )

        return all_questions

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
            from src.db.repositories.audit_repository import find_cached_generation
            from src.agents.core.audit_queue import enqueue_audit_event

            async def check_cache():
                return await find_cached_generation(
                    document_name=document_name,
                    generation_type='questions',
                    model=self.config.openai_model,
                    content_hash=content_hash,
                    organization_id=organization_id
                )

            cached = run_async(check_cache())
            if cached and cached.get('content', {}).get('questions'):
                cache_duration_ms = elapsed_ms(start_time)
                cached_questions = cached['content']['questions']
                logger.info(f"Cache hit for questions: {document_name} ({cache_duration_ms:.1f}ms)")

                # Log cache hit via audit queue (non-blocking)
                enqueue_audit_event(
                    event_type='generation_cache_hit',
                    file_name=document_name,
                    organization_id=organization_id,
                    document_hash=content_hash,
                    details={
                        'generation_type': 'questions',
                        'generation_id': str(cached['id']),
                        'content_hash': content_hash
                    }
                )

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

            # Log generation start via audit queue (non-blocking)
            enqueue_audit_event(
                event_type='generation_started',
                file_name=document_name,
                organization_id=organization_id,
                document_hash=content_hash,
                details={
                    'generation_type': 'questions',
                    'content_hash': content_hash,
                    'num_questions': num_questions
                }
            )

        except ImportError:
            logger.debug("Audit module not available, skipping cache check")
        except Exception as e:
            logger.warning(f"Cache check failed, proceeding with generation: {e}")

        # Use parallel generation for better performance
        try:
            questions = self._generate_parallel(content, num_questions)

            duration_ms = elapsed_ms(start_time)
            logger.info(f"Generated {len(questions)} questions in {duration_ms:.1f}ms")

            # Count by difficulty
            difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
            for q in questions:
                d = q.get("difficulty", "medium").lower()
                if d in difficulty_counts:
                    difficulty_counts[d] += 1

            # Log generation complete via audit queue (non-blocking)
            try:
                from src.agents.core.audit_queue import enqueue_audit_event

                enqueue_audit_event(
                    event_type='generation_completed',
                    file_name=document_name,
                    organization_id=organization_id,
                    document_hash=content_hash,
                    details={
                        'generation_type': 'questions',
                        'content_hash': content_hash,
                        'count': len(questions),
                        'difficulty_distribution': difficulty_counts,
                        'processing_time_ms': duration_ms
                    }
                )
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
