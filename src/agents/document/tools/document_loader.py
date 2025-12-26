"""Document loader tool for loading content from GCS or local storage."""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from ..config import DocumentAgentConfig
from .base import DocumentLoaderInput
from src.utils.async_utils import run_async
from src.utils.timer_utils import elapsed_ms

logger = logging.getLogger(__name__)


class DocumentLoaderTool(BaseTool):
    """Tool to load document content from GCS storage or local upload directory."""

    name: str = "document_loader"
    description: str = """Load document content from storage.
    Uses the provided parsed_file_path to load from GCS,
    then falls back to local /upload directory for raw text files (.txt, .md).
    Returns the document content and source path."""
    args_schema: Type[BaseModel] = DocumentLoaderInput

    config: DocumentAgentConfig = Field(default_factory=DocumentAgentConfig)

    def _run(
        self,
        document_name: str,
        parsed_file_path: str = "",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Load document content from GCS or local storage."""
        start_time = time.time()

        async def load_from_gcs():
            """Try loading from GCS using parsed_file_path directly."""
            from src.storage import get_storage

            storage = get_storage()

            # Use parsed_file_path directly with use_prefix=False to bypass demo_docs prefix
            if await storage.exists(parsed_file_path, use_prefix=False):
                content = await storage.read(parsed_file_path, use_prefix=False)
                if content:
                    return {
                        "success": True,
                        "content": content,
                        "source_path": f"gs://{storage.bucket_name}/{parsed_file_path}",
                        "source_type": "parsed_gcs",
                        "parsed_file_path": parsed_file_path,
                        "content_length": len(content)
                    }

            return None

        # Try GCS first with provided parsed_file_path
        if parsed_file_path:
            try:
                result = run_async(load_from_gcs())
                if result:
                    duration_ms = elapsed_ms(start_time)
                    logger.info(f"Loaded document from GCS: {parsed_file_path} ({result['content_length']} chars, {duration_ms:.1f}ms)")
                    return json.dumps(result)
            except Exception as e:
                logger.warning(f"GCS read failed for {parsed_file_path}, falling back to local: {e}")

        # Fallback to local upload directory for raw text files
        base_path = Path(os.getcwd())
        upload_dir = base_path / self.config.upload_directory

        for ext in ['.md', '.txt']:
            # Try exact name first
            upload_path = upload_dir / document_name
            if upload_path.exists() and upload_path.suffix.lower() in ['.md', '.txt']:
                content = upload_path.read_text(encoding='utf-8')
                duration_ms = elapsed_ms(start_time)
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
                duration_ms = elapsed_ms(start_time)
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
        error_msg = f"Document '{document_name}' not found in GCS storage ({parsed_file_path}) or upload ({upload_dir}) directory"
        logger.warning(error_msg)
        return json.dumps({
            "success": False,
            "error": error_msg,
            "searched_paths": [parsed_file_path, str(upload_dir)]
        })
