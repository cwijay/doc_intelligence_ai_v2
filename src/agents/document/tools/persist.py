"""Content persistence tool for saving generated content to GCS and PostgreSQL."""

import asyncio
import json
import logging
import time
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import (
    ContentPersistInput,
    build_content_path,
    format_summary_markdown,
    format_faqs_json,
    format_questions_json,
)
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


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
        parsed_file_path: str = "",
        summary: Optional[str] = None,
        faqs: Optional[str] = None,
        questions: Optional[str] = None,
        content_hash: Optional[str] = None,
        organization_id: Optional[str] = None,
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
            "parsed_file_path": parsed_file_path,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": summary,
            "faqs": faqs_list,
            "questions": questions_list,
            "model": self.config.gemini_model
        }

        results = {
            "success": True,
            "document_name": document_name,
            "parsed_file_path": parsed_file_path,
            "database_saved": False,
            "file_saved": False
        }

        # Save to GCS - separate files for each content type
        saved_paths = {}
        try:
            from src.storage import get_storage

            storage = get_storage()
            generated_at = content["generated_at"]

            async def save_content_files():
                """Save each content type to its own file in separate directories."""
                paths = {}

                # Save summary as markdown
                if summary:
                    summary_path = build_content_path(parsed_file_path, "summary", document_name)
                    if '/' in summary_path:
                        summary_dir, summary_filename = summary_path.rsplit('/', 1)
                    else:
                        summary_dir, summary_filename = "", summary_path

                    summary_content = format_summary_markdown(
                        summary=summary,
                        document_name=document_name,
                        model=self.config.gemini_model,
                        generated_at=generated_at,
                        content_hash=content_hash
                    )

                    uri = await storage.save(
                        summary_content,
                        summary_filename,
                        directory=summary_dir,
                        use_prefix=False
                    )
                    paths["summary"] = uri
                    logger.info(f"Saved summary to GCS: {uri}")

                # Save FAQs as JSON
                if faqs_list:
                    faqs_path = build_content_path(parsed_file_path, "faqs", document_name)
                    if '/' in faqs_path:
                        faqs_dir, faqs_filename = faqs_path.rsplit('/', 1)
                    else:
                        faqs_dir, faqs_filename = "", faqs_path

                    faqs_content = format_faqs_json(
                        faqs=faqs_list,
                        document_name=document_name,
                        parsed_file_path=parsed_file_path,
                        model=self.config.gemini_model,
                        generated_at=generated_at,
                        content_hash=content_hash
                    )

                    uri = await storage.save(
                        faqs_content,
                        faqs_filename,
                        directory=faqs_dir,
                        use_prefix=False
                    )
                    paths["faqs"] = uri
                    logger.info(f"Saved FAQs to GCS: {uri}")

                # Save questions as JSON
                if questions_list:
                    questions_path = build_content_path(parsed_file_path, "questions", document_name)
                    if '/' in questions_path:
                        questions_dir, questions_filename = questions_path.rsplit('/', 1)
                    else:
                        questions_dir, questions_filename = "", questions_path

                    questions_content = format_questions_json(
                        questions=questions_list,
                        document_name=document_name,
                        parsed_file_path=parsed_file_path,
                        model=self.config.gemini_model,
                        generated_at=generated_at,
                        content_hash=content_hash
                    )

                    uri = await storage.save(
                        questions_content,
                        questions_filename,
                        directory=questions_dir,
                        use_prefix=False
                    )
                    paths["questions"] = uri
                    logger.info(f"Saved questions to GCS: {uri}")

                return paths

            saved_paths = run_async(save_content_files())

            if saved_paths:
                results["file_saved"] = True
                results["output_file_paths"] = saved_paths
                # For backward compatibility, set output_file_path to first saved path
                results["output_file_path"] = next(iter(saved_paths.values()))
                logger.info(f"Saved generated content to GCS: {len(saved_paths)} files")

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

                # Build options dict with GCS paths
                options = {
                    "generated_at": content["generated_at"],
                    "gcs_paths": saved_paths if saved_paths else {},
                }

                processing_time = elapsed_ms(start_time)

                # Run async function
                async def save_async():
                    return await save_document_generation(
                        document_name=document_name,
                        source_path=parsed_file_path,  # GCS path to parsed document
                        generation_type=generation_type,
                        content=content_for_db,
                        options=options,
                        model=self.config.gemini_model,
                        processing_time_ms=processing_time,
                        document_hash=content_hash,  # Pass content hash for cache validation
                        organization_id=organization_id,
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

        duration_ms = elapsed_ms(start_time)
        results["processing_time_ms"] = duration_ms

        return json.dumps(results)
