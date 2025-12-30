"""LangChain tools for Document Agent."""

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .config import DocumentAgentConfig
from .schemas import FAQ, Question

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Helper to run async code from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# Tool Input Schemas
class DocumentLoaderInput(BaseModel):
    """Input for document loader tool."""
    document_name: str = Field(description="Name of the document to load (e.g., 'Sample1.md')")
    org_name: str = Field(description="Organization name for GCS path (e.g., 'ACME corp')")
    folder_name: str = Field(description="Folder name within organization (e.g., 'invoices')")


class SummaryGeneratorInput(BaseModel):
    """Input for summary generator tool."""
    content: str = Field(description="Document content to summarize")
    document_name: str = Field(description="Name of the source document for caching")
    max_words: int = Field(default=500, description="Maximum words for summary")


class FAQGeneratorInput(BaseModel):
    """Input for FAQ generator tool."""
    content: str = Field(description="Document content to generate FAQs from")
    document_name: str = Field(description="Name of the source document for caching")
    num_faqs: int = Field(default=5, description="Number of FAQs to generate")


class QuestionGeneratorInput(BaseModel):
    """Input for question generator tool."""
    content: str = Field(description="Document content to generate questions from")
    document_name: str = Field(description="Name of the source document for caching")
    num_questions: int = Field(default=10, description="Number of questions to generate")


class ContentPersistInput(BaseModel):
    """Input for content persist tool."""
    document_name: str = Field(description="Name of the source document")
    org_name: str = Field(description="Organization name for GCS path (e.g., 'ACME corp')")
    folder_name: str = Field(description="Folder name within organization (e.g., 'invoices')")
    summary: Optional[str] = Field(default=None, description="Generated summary")
    faqs: Optional[str] = Field(default=None, description="Generated FAQs as JSON string")
    questions: Optional[str] = Field(default=None, description="Generated questions as JSON string")
    content_hash: Optional[str] = Field(default=None, description="SHA-256 hash of source document content for cache invalidation")


class DocumentLoaderTool(BaseTool):
    """Tool to load document content from GCS storage or local upload directory."""

    name: str = "document_loader"
    description: str = """Load document content from storage.
    Searches GCS parsed directory first for pre-parsed .md files,
    then falls back to local /upload directory for raw text files (.txt, .md).
    Returns the document content and source path."""
    args_schema: Type[BaseModel] = DocumentLoaderInput

    config: DocumentAgentConfig = Field(default_factory=DocumentAgentConfig)

    def _run(
        self,
        document_name: str,
        org_name: str = "",
        folder_name: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Load document content from GCS or local storage."""
        start_time = time.time()

        async def load_from_gcs():
            """Try loading from GCS parsed directory."""
            from src.storage import get_storage, get_storage_config

            storage = get_storage()
            config = get_storage_config()

            # Ensure .md extension for parsed files
            if not document_name.endswith('.md'):
                md_name = Path(document_name).stem + ".md"
            else:
                md_name = document_name

            # Build path: {org_name}/parsed/{folder_name}/filename.md
            parsed_path = f"{org_name}/parsed/{folder_name}/{md_name}"

            # Try to read from GCS
            if await storage.exists(parsed_path, use_prefix=False):
                content = await storage.read(parsed_path, use_prefix=False)
                if content:
                    return {
                        "success": True,
                        "content": content,
                        "source_path": storage.get_uri(parsed_path),
                        "source_type": "parsed_gcs",
                        "org_name": org_name,
                        "folder_name": folder_name,
                        "content_length": len(content)
                    }

            return None

        # Try GCS first
        try:
            result = _run_async(load_from_gcs())
            if result:
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Loaded document from GCS: {document_name} ({result['content_length']} chars, {duration_ms:.1f}ms)")
                return json.dumps(result)
        except Exception as e:
            logger.warning(f"GCS read failed, falling back to local: {e}")

        # Fallback to local upload directory for raw text files
        base_path = Path(os.getcwd())
        upload_dir = base_path / self.config.upload_directory

        for ext in ['.md', '.txt']:
            # Try exact name first
            upload_path = upload_dir / document_name
            if upload_path.exists() and upload_path.suffix.lower() in ['.md', '.txt']:
                content = upload_path.read_text(encoding='utf-8')
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Loaded document from upload: {document_name} ({len(content)} chars, {duration_ms:.1f}ms)")
                return json.dumps({
                    "success": True,
                    "content": content,
                    "source_path": str(upload_path),
                    "source_type": "upload",
                    "file_size_bytes": upload_path.stat().st_size,
                    "content_length": len(content)
                })

            # Try with extension added
            upload_path_ext = upload_dir / f"{Path(document_name).stem}{ext}"
            if upload_path_ext.exists():
                content = upload_path_ext.read_text(encoding='utf-8')
                duration_ms = (time.time() - start_time) * 1000
                logger.info(f"Loaded document from upload: {upload_path_ext.name} ({len(content)} chars, {duration_ms:.1f}ms)")
                return json.dumps({
                    "success": True,
                    "content": content,
                    "source_path": str(upload_path_ext),
                    "source_type": "upload",
                    "file_size_bytes": upload_path_ext.stat().st_size,
                    "content_length": len(content)
                })

        # Document not found
        error_msg = f"Document '{document_name}' not found in GCS storage or upload ({upload_dir}) directory"
        logger.warning(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "searched_paths": [str(upload_dir)]
        })


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
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate summary from document content."""
        start_time = time.time()

        # Compute content hash for cache validation
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check cache before generating
        try:
            from src.db.repositories.audit_repository import find_cached_generation, log_event

            async def check_cache():
                cached = await find_cached_generation(
                    document_name=document_name,
                    generation_type='summary',
                    model=self.config.gemini_model,
                    content_hash=content_hash
                )
                if cached:
                    await log_event(
                        event_type='generation_cache_hit',
                        file_name=document_name,
                        details={
                            'generation_type': 'summary',
                            'generation_id': str(cached['id']),
                            'content_hash': content_hash
                        }
                    )
                return cached

            cached = _run_async(check_cache())
            if cached and cached.get('content', {}).get('summary'):
                cache_duration_ms = (time.time() - start_time) * 1000
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
                    details={
                        'generation_type': 'summary',
                        'content_hash': content_hash,
                        'max_words': max_words
                    }
                )
            _run_async(log_start())

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
            summary = response.content.strip()

            duration_ms = (time.time() - start_time) * 1000
            word_count = len(summary.split())
            logger.info(f"Generated summary: {word_count} words in {duration_ms:.1f}ms")

            # Log generation complete
            try:
                from src.db.repositories.audit_repository import log_event

                async def log_complete():
                    await log_event(
                        event_type='generation_completed',
                        file_name=document_name,
                        details={
                            'generation_type': 'summary',
                            'content_hash': content_hash,
                            'word_count': word_count,
                            'processing_time_ms': duration_ms
                        }
                    )
                _run_async(log_complete())
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
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate FAQs from document content."""
        start_time = time.time()

        # Compute content hash for cache validation
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check cache before generating
        try:
            from src.db.repositories.audit_repository import find_cached_generation, log_event

            async def check_cache():
                cached = await find_cached_generation(
                    document_name=document_name,
                    generation_type='faqs',
                    model=self.config.gemini_model,
                    content_hash=content_hash
                )
                if cached:
                    await log_event(
                        event_type='generation_cache_hit',
                        file_name=document_name,
                        details={
                            'generation_type': 'faqs',
                            'generation_id': str(cached['id']),
                            'content_hash': content_hash
                        }
                    )
                return cached

            cached = _run_async(check_cache())
            if cached and cached.get('content', {}).get('faqs'):
                cache_duration_ms = (time.time() - start_time) * 1000
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
                    details={
                        'generation_type': 'faqs',
                        'content_hash': content_hash,
                        'num_faqs': num_faqs
                    }
                )
            _run_async(log_start())

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
            response_text = response.content.strip()

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_text)
            faqs = parsed.get("faqs", [])

            duration_ms = (time.time() - start_time) * 1000
            logger.info(f"Generated {len(faqs)} FAQs in {duration_ms:.1f}ms")

            # Log generation complete
            try:
                from src.db.repositories.audit_repository import log_event

                async def log_complete():
                    await log_event(
                        event_type='generation_completed',
                        file_name=document_name,
                        details={
                            'generation_type': 'faqs',
                            'content_hash': content_hash,
                            'count': len(faqs),
                            'processing_time_ms': duration_ms
                        }
                    )
                _run_async(log_complete())
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
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Generate comprehension questions from document content."""
        start_time = time.time()

        # Compute content hash for cache validation
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Check cache before generating
        try:
            from src.db.repositories.audit_repository import find_cached_generation, log_event

            async def check_cache():
                cached = await find_cached_generation(
                    document_name=document_name,
                    generation_type='questions',
                    model=self.config.gemini_model,
                    content_hash=content_hash
                )
                if cached:
                    await log_event(
                        event_type='generation_cache_hit',
                        file_name=document_name,
                        details={
                            'generation_type': 'questions',
                            'generation_id': str(cached['id']),
                            'content_hash': content_hash
                        }
                    )
                return cached

            cached = _run_async(check_cache())
            if cached and cached.get('content', {}).get('questions'):
                cache_duration_ms = (time.time() - start_time) * 1000
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
                    details={
                        'generation_type': 'questions',
                        'content_hash': content_hash,
                        'num_questions': num_questions
                    }
                )
            _run_async(log_start())

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
            response_text = response.content.strip()

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(response_text)
            questions = parsed.get("questions", [])

            duration_ms = (time.time() - start_time) * 1000
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
                        details={
                            'generation_type': 'questions',
                            'content_hash': content_hash,
                            'count': len(questions),
                            'difficulty_distribution': difficulty_counts,
                            'processing_time_ms': duration_ms
                        }
                    )
                _run_async(log_complete())
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


class ContentPersistTool(BaseTool):
    """Tool to persist generated content to PostgreSQL and GCS."""

    name: str = "content_persist"
    description: str = """Save generated content (summary, FAQs, questions) to storage.
    Persists to PostgreSQL for audit trail and to GCS as JSON files.
    Returns persistence status and file paths."""
    args_schema: Type[BaseModel] = ContentPersistInput

    config: DocumentAgentConfig = Field(default_factory=DocumentAgentConfig)

    def _run(
        self,
        document_name: str,
        org_name: str = "",
        folder_name: str = "",
        summary: Optional[str] = None,
        faqs: Optional[str] = None,
        questions: Optional[str] = None,
        content_hash: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Persist generated content to GCS and PostgreSQL."""
        start_time = time.time()

        # Parse JSON strings back to objects
        faqs_list = json.loads(faqs) if faqs else None
        questions_list = json.loads(questions) if questions else None

        # Prepare content object
        content = {
            "document_name": document_name,
            "org_name": org_name,
            "folder_name": folder_name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": summary,
            "faqs": faqs_list,
            "questions": questions_list,
            "model": self.config.gemini_model
        }

        results = {
            "success": True,
            "document_name": document_name,
            "org_name": org_name,
            "folder_name": folder_name,
            "database_saved": False,
            "file_saved": False
        }

        # Save to GCS
        try:
            from src.storage import get_storage, get_storage_config

            storage = get_storage()
            storage_config = get_storage_config()

            output_filename = f"{Path(document_name).stem}_generated.json"
            json_content = json.dumps(content, indent=2, ensure_ascii=False)

            # Build directory path: {org_name}/generated/{folder_name}
            generated_directory = f"{org_name}/generated/{folder_name}"

            async def save_to_gcs():
                return await storage.save(
                    json_content,
                    output_filename,
                    directory=generated_directory
                )

            gcs_uri = _run_async(save_to_gcs())

            results["file_saved"] = True
            results["output_file_path"] = gcs_uri
            logger.info(f"Saved generated content to GCS: {gcs_uri}")

        except Exception as e:
            logger.error(f"Failed to save to GCS: {e}")
            results["file_error"] = str(e)

        # Save to PostgreSQL if enabled
        if self.config.persist_to_database:
            try:
                from src.db.repositories.audit_repository import save_document_generation

                # Determine generation type
                if summary and faqs_list and questions_list:
                    generation_type = "all"
                elif summary:
                    generation_type = "summary"
                elif faqs_list:
                    generation_type = "faqs"
                elif questions_list:
                    generation_type = "questions"
                else:
                    generation_type = "all"

                # Build content dict for storage
                content_for_db = {
                    "summary": summary,
                    "faqs": faqs_list,
                    "questions": questions_list,
                }

                # Build options dict
                options = {
                    "generated_at": content["generated_at"],
                }

                processing_time = (time.time() - start_time) * 1000

                # Run async function
                async def save_async():
                    return await save_document_generation(
                        document_name=document_name,
                        source_path="",  # Not available in this context
                        generation_type=generation_type,
                        content=content_for_db,
                        options=options,
                        model=self.config.gemini_model,
                        processing_time_ms=processing_time,
                        document_hash=content_hash,  # Pass content hash for cache validation
                    )

                # Get or create event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If we're already in an async context, create a task
                        import nest_asyncio
                        nest_asyncio.apply()
                        doc_id = loop.run_until_complete(save_async())
                    else:
                        doc_id = loop.run_until_complete(save_async())
                except RuntimeError:
                    # No event loop, create one
                    doc_id = asyncio.run(save_async())

                results["database_saved"] = True
                results["database_id"] = doc_id
                logger.info(f"Saved to PostgreSQL: {doc_id}")

            except ImportError as e:
                logger.warning(f"Database module not available: {e}")
                results["database_error"] = f"Database module not available: {e}"
            except Exception as e:
                logger.error(f"Failed to save to database: {e}")
                results["database_error"] = str(e)

        duration_ms = (time.time() - start_time) * 1000
        results["processing_time_ms"] = duration_ms

        return json.dumps(results)


def create_document_tools(config: DocumentAgentConfig) -> List[BaseTool]:
    """Create all document processing tools with shared config."""
    return [
        DocumentLoaderTool(config=config),
        SummaryGeneratorTool(config=config),
        FAQGeneratorTool(config=config),
        QuestionGeneratorTool(config=config),
        ContentPersistTool(config=config)
    ]
