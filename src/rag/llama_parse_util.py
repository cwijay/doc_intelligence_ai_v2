"""
LlamaParse Document Parser Utility

This module provides functionality to parse various document formats
using LlamaIndex's LlamaParse and output markdown with proper table
and image recognition, including OCR for handwritten content.

Supported formats: PDF, JPEG, JPG, PNG, DOCX, PPTX, TXT, XLSX, XLS, and more.
"""

import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Suppress httpx/llama noise BEFORE import
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_cloud_services").setLevel(logging.WARNING)

# Get logger (config should be set by entry point)
logger = logging.getLogger(__name__)

# Now import (httpx will respect the logging level)
from llama_cloud_services import LlamaParse

# Load environment variables
load_dotenv()

# Supported file extensions (set for O(1) lookup)
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf",
    ".doc", ".docx", ".docm",
    ".ppt", ".pptx", ".pptm",
    ".txt", ".rtf",
    # Spreadsheets
    ".xlsx", ".xls", ".xlsm", ".xlsb",
    ".csv", ".tsv",
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp",
    # Web
    ".html", ".htm",
}

# Configuration
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
DEFAULT_LANGUAGE = os.getenv("LLAMA_PARSE_LANGUAGE", "en")
LLAMA_PARSE_LLM = os.getenv("LLAMA_PARSE_LLM", "gemini-2.5-pro")
AGENT_MULTIMODAL_MODEL = os.getenv("AGENT_MULTIMODAL_MODEL", "openai-gpt-5-mini")
# External provider API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def _get_parser() -> LlamaParse:
    """
    Create and return a configured LlamaParse instance.

    Configuration includes:
    - Multimodal LLM for image/handwriting understanding
    - Enhanced table extraction with HTML support
    - Layout preservation across pages
    - Screenshot capture for visual analysis
    - Custom prompts for improved handwriting OCR accuracy

    Returns:
        LlamaParse: Configured parser instance

    Raises:
        ValueError: If LLAMA_CLOUD_API_KEY is not set
    """
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError(
            "LLAMA_CLOUD_API_KEY environment variable is not set. "
            "Get your API key from https://cloud.llamaindex.ai/parse"
        )

    # Generic prompts for handwritten document handling
    user_prompt = """This document contains handwritten entries mixed with printed text.
Pay careful attention to:
- Handwritten alphanumeric codes (lot numbers, reference numbers, dates)
- Handwritten measurements and numeric values
- Checkmarks, initials, and signatures
- Table cells with handwritten content
Ensure all handwritten text is accurately transcribed."""

    system_prompt_append = """When parsing documents with handwritten content:
- Keep table columns properly aligned even when cells contain handwritten entries
- Preserve the distinction between printed headers and handwritten values
- Recognize checkmarks (√, ✓) as verification marks
- Maintain numbered list formatting for procedures and steps
- Transcribe handwritten numbers accurately, distinguishing similar characters (0/O, 1/l, 5/S)"""

    return LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        language=DEFAULT_LANGUAGE,

        # Prompts for improved handwriting accuracy
        user_prompt=user_prompt,
        system_prompt_append=system_prompt_append,

        # Multimodal LLM for document parsing
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name=LLAMA_PARSE_LLM,

        # Table extraction
        output_tables_as_HTML=True,  # Better table structure with colspan/rowspan

        # Visual content
        take_screenshot=True,  # Capture page screenshots for visual analysis
    )


def _get_agent_parser(complexity: str = "normal") -> LlamaParse:
    """
    Create a LlamaParse instance for agent mode parsing.

    Args:
        complexity: Document complexity level
                   - "normal": uses openai-gpt-5-mini (default, cost-effective)
                   - "high": uses gemini-2.5-pro (better for complex handwriting)
    """
    if not LLAMA_CLOUD_API_KEY:
        raise ValueError("LLAMA_CLOUD_API_KEY not set")

    # Select model and API key based on complexity
    if complexity == "high":
        model = "gemini-2.5-pro"
        api_key = GOOGLE_API_KEY
    else:
        model = "openai-gpt-5-mini"
        api_key = OPENAI_API_KEY

    if not api_key:
        raise ValueError(f"API key required for {model}")

    return LlamaParse(
        api_key=LLAMA_CLOUD_API_KEY,
        language=DEFAULT_LANGUAGE,
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name=model,
        vendor_multimodal_api_key=api_key,
        output_tables_as_HTML=True,
        take_screenshot=True,
    )


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
            f"Unsupported file format: {path.suffix}. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return path


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


def _save_markdown(content: str, file_path: Path, output_dir: str) -> str:
    """
    Sync wrapper for saving markdown to GCS storage.

    Args:
        content: Markdown content to save
        file_path: Original file path (used for naming)
        output_dir: Directory within storage to save the file

    Returns:
        str: GCS URI of the saved markdown file
    """
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - use nest_asyncio pattern
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(
            _save_markdown_async(content, file_path, output_dir)
        )
    except RuntimeError:
        # No running loop - create new one
        return asyncio.run(_save_markdown_async(content, file_path, output_dir))


def parse_document(
    file_path: str,
    output_dir: str = None,
    use_agent_mode: bool = False,
    complexity: str = "normal"
) -> str:
    """
    Parse a document and return markdown content.

    Args:
        file_path: Path to the document to parse
        output_dir: Optional directory to save the markdown file.
        use_agent_mode: Use agent parsing mode for difficult documents.
        complexity: Document complexity ("normal" or "high")
                   - "normal": openai-gpt-5-mini (default)
                   - "high": gemini-2.5-pro

    Returns:
        str: Parsed document content in markdown format

    Example:
        >>> content = parse_document("invoice.pdf")
        >>> content = parse_document("handwritten.pdf", use_agent_mode=True)
        >>> content = parse_document("complex.pdf", use_agent_mode=True, complexity="high")
    """
    path = _validate_file(file_path)
    logger.info(f"Parsing document: {path.name} (agent_mode={use_agent_mode}, complexity={complexity})")

    parser = _get_agent_parser(complexity) if use_agent_mode else _get_parser()

    try:
        # Parse the document - returns list of Document objects
        documents = parser.load_data(str(path))

        if not documents:
            logger.warning(f"No content extracted from: {path.name}")
            return ""

        # Extract markdown content from each document
        content = "\n\n---\n\n".join(doc.text for doc in documents)

        logger.info(f"Successfully parsed: {path.name}")

        # Save to file if output_dir is specified
        if output_dir:
            _save_markdown(content, path, output_dir)

        return content

    except Exception as e:
        logger.error(f"Error parsing {path.name}: {e}")
        raise


def parse_documents(file_paths: list[str], output_dir: str = None) -> list[str]:
    """
    Parse multiple documents and return list of markdown contents.

    Args:
        file_paths: List of paths to documents to parse
        output_dir: Optional directory to save markdown files

    Returns:
        list[str]: List of parsed document contents in markdown format

    Raises:
        FileNotFoundError: If any file doesn't exist
        ValueError: If any file extension is not supported

    Example:
        >>> contents = parse_documents(["doc1.pdf", "doc2.docx"])
        >>> for i, content in enumerate(contents):
        ...     print(f"Document {i+1}: {len(content)} characters")
    """
    # Validate all files first
    paths = [_validate_file(fp) for fp in file_paths]
    logger.info(f"Parsing {len(paths)} documents")

    parser = _get_parser()

    try:
        # Parse each document individually
        contents = []
        for i, path in enumerate(paths):
            documents = parser.load_data(str(path))
            content = "\n\n---\n\n".join(doc.text for doc in documents)
            contents.append(content)

            # Save to file if output_dir is specified
            if output_dir:
                _save_markdown(content, path, output_dir)

        logger.info(f"Successfully parsed {len(contents)} documents")
        return contents

    except Exception as e:
        logger.error(f"Error parsing documents: {e}")
        raise


def get_supported_extensions() -> list[str]:
    """
    Return list of supported file extensions.

    Returns:
        list[str]: List of supported file extensions (e.g., ['.pdf', '.docx', ...])

    Example:
        >>> extensions = get_supported_extensions()
        >>> print(extensions)
        ['.pdf', '.doc', '.docx', ...]
    """
    return sorted(SUPPORTED_EXTENSIONS)
