"""
Gemini Document Parser Utility

This module provides functionality to parse various document formats
using Google Gemini models and output markdown with proper table
and image recognition, including OCR for handwritten content.

Alternative to LlamaParse with native async support and faster processing.

Supported formats: PDF, JPEG, JPG, PNG, GIF, BMP, TIFF, WEBP
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)
load_dotenv()

# Supported file extensions (Gemini File API supports these)
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    # Images (for OCR/visual parsing)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
}

# MIME type mapping
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configuration
GEMINI_PARSE_MODEL = os.getenv("GEMINI_PARSE_MODEL", "gemini-3-flash-preview")
GEMINI_PARSE_MODEL_HIGH = os.getenv("GEMINI_PARSE_MODEL_HIGH", "gemini-3-pro-preview")
GEMINI_PARSE_TIMEOUT = int(os.getenv("GEMINI_PARSE_TIMEOUT_SECONDS", "120"))

# System prompt for document parsing
DOCUMENT_PARSE_PROMPT = """You are a document parser. Convert this document to well-formatted Markdown.

Requirements:
1. Preserve ALL text content accurately - do not summarize or omit any text
2. Format tables using proper Markdown table syntax (| header | header |)
3. Use appropriate heading levels (# ## ###) based on document structure
4. Preserve lists (bulleted and numbered)
5. Maintain logical document flow and sections
6. For any measurements, values, or data, ensure accuracy
7. If there are images or diagrams, describe them briefly in [Image: description] format
8. Use **bold** for emphasis where appropriate (e.g., key findings, important values)
9. Preserve any form fields, checkboxes, or interactive elements as text

Output ONLY the Markdown content, no explanations or preamble."""

# Enhanced prompt for handwritten documents
HANDWRITTEN_PARSE_PROMPT = """You are a document parser specializing in handwritten content. Convert this document to well-formatted Markdown.

Requirements:
1. Preserve ALL text content accurately, including handwritten entries
2. Pay careful attention to:
   - Handwritten alphanumeric codes (lot numbers, reference numbers, dates)
   - Handwritten measurements and numeric values
   - Checkmarks (transcribe as [✓] or [X]), initials, and signatures
   - Table cells with handwritten content
3. Format tables using proper Markdown table syntax (| header | header |)
4. Keep table columns properly aligned even when cells contain handwritten entries
5. Preserve the distinction between printed headers and handwritten values
6. Transcribe handwritten numbers accurately, distinguishing similar characters (0/O, 1/l, 5/S)
7. Use appropriate heading levels (# ## ###) based on document structure
8. Maintain numbered list formatting for procedures and steps

Output ONLY the Markdown content, no explanations or preamble."""


def _get_client() -> genai.Client:
    """Get or create Gemini client."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Get your API key from https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=GOOGLE_API_KEY)


def _validate_file(file_path: str) -> Path:
    """
    Validate that the file exists and has a supported extension.

    Args:
        file_path: Path to the file to validate

    Returns:
        Path: Validated Path object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file extension is not supported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format for Gemini parser: {path.suffix}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    return path


def _get_mime_type(file_path: Path) -> str:
    """Get MIME type for file."""
    return MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")


async def _upload_file_async(client: genai.Client, file_path: Path) -> types.File:
    """Upload file to Gemini asynchronously."""
    loop = asyncio.get_running_loop()

    def _upload():
        return client.files.upload(file=str(file_path))

    return await loop.run_in_executor(None, _upload)


async def _delete_file_async(client: genai.Client, file_name: str) -> None:
    """Delete uploaded file from Gemini asynchronously."""
    loop = asyncio.get_running_loop()

    def _delete():
        try:
            client.files.delete(name=file_name)
        except Exception as e:
            logger.warning(f"Failed to delete uploaded file {file_name}: {e}")

    await loop.run_in_executor(None, _delete)


async def _generate_content_async(
    client: genai.Client,
    model: str,
    uploaded_file: types.File,
    prompt: str,
    mime_type: str,
) -> str:
    """Generate content using Gemini model asynchronously."""
    loop = asyncio.get_running_loop()

    def _generate():
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=mime_type),
                        types.Part(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temperature for accurate transcription
            )
        )
        return response.text

    return await loop.run_in_executor(None, _generate)


async def parse_document_async(
    file_path: str,
    output_dir: Optional[str] = None,
    use_handwriting_mode: bool = False,
    complexity: str = "normal"
) -> str:
    """
    Async: Parse a document using Gemini and return markdown content.

    Args:
        file_path: Path to the document to parse
        output_dir: Optional directory to save the markdown file
        use_handwriting_mode: Use enhanced prompts for handwritten documents
        complexity: Document complexity ("normal" or "high")
                   - "normal": gemini-2.0-flash (default, fast)
                   - "high": gemini-2.5-pro (better for complex documents)

    Returns:
        str: Parsed document content in markdown format

    Example:
        >>> content = await parse_document_async("invoice.pdf")
        >>> content = await parse_document_async("handwritten.pdf", use_handwriting_mode=True)
        >>> content = await parse_document_async("complex.pdf", complexity="high")
    """
    path = _validate_file(file_path)
    mime_type = _get_mime_type(path)
    model = GEMINI_PARSE_MODEL_HIGH if complexity == "high" else GEMINI_PARSE_MODEL
    prompt = HANDWRITTEN_PARSE_PROMPT if use_handwriting_mode else DOCUMENT_PARSE_PROMPT

    logger.info(f"Parsing document (Gemini async): {path.name} (model={model}, handwriting={use_handwriting_mode})")

    client = _get_client()
    uploaded_file = None
    start_time = time.time()

    try:
        # Upload file to Gemini
        logger.debug(f"Uploading file to Gemini: {path.name}")
        uploaded_file = await _upload_file_async(client, path)
        logger.debug(f"File uploaded: {uploaded_file.name}")

        # Generate markdown content
        content = await _generate_content_async(
            client, model, uploaded_file, prompt, mime_type
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Successfully parsed (Gemini): {path.name} in {elapsed_ms}ms")

        if not content:
            logger.warning(f"No content extracted from: {path.name}")
            return ""

        # Save to file if output_dir is specified
        if output_dir:
            await _save_markdown_async(content, path, output_dir)

        return content

    except Exception as e:
        logger.error(f"Error parsing {path.name} with Gemini: {e}")
        raise

    finally:
        # Clean up uploaded file
        if uploaded_file:
            await _delete_file_async(client, uploaded_file.name)


def parse_document(
    file_path: str,
    output_dir: Optional[str] = None,
    use_handwriting_mode: bool = False,
    complexity: str = "normal"
) -> str:
    """
    Sync: Parse a document using Gemini and return markdown content.

    Args:
        file_path: Path to the document to parse
        output_dir: Optional directory to save the markdown file
        use_handwriting_mode: Use enhanced prompts for handwritten documents
        complexity: Document complexity ("normal" or "high")
                   - "normal": gemini-2.0-flash (default, fast)
                   - "high": gemini-2.5-pro (better for complex documents)

    Returns:
        str: Parsed document content in markdown format

    Example:
        >>> content = parse_document("invoice.pdf")
        >>> content = parse_document("handwritten.pdf", use_handwriting_mode=True)
    """
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - use nest_asyncio pattern
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(
            parse_document_async(file_path, output_dir, use_handwriting_mode, complexity)
        )
    except RuntimeError:
        # No running loop - create new one
        return asyncio.run(
            parse_document_async(file_path, output_dir, use_handwriting_mode, complexity)
        )


async def _save_markdown_async(content: str, file_path: Path, output_dir: str) -> str:
    """
    Save markdown content to GCS storage.

    Args:
        content: Markdown content to save
        file_path: Original file path (used for naming)
        output_dir: Directory within storage to save the file

    Returns:
        str: GCS URI of the saved markdown file
    """
    from src.storage import get_storage

    storage = get_storage()
    md_filename = file_path.stem + ".md"

    # Save to GCS storage
    uri = await storage.save(content, md_filename, directory=output_dir)
    logger.info(f"Saved markdown to: {uri}")
    return uri


def get_supported_extensions() -> list[str]:
    """
    Return list of supported file extensions for Gemini parser.

    Returns:
        list[str]: List of supported file extensions
    """
    return sorted(SUPPORTED_EXTENSIONS)
