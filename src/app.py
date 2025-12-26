"""Test client for Gemini File Store - Parse, Upload, and Search."""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_gemini_file_search():
    """Test parsing a PDF, uploading to Gemini file store, and searching."""
    from src.rag.llama_parse_util import parse_document
    from src.rag.gemini_file_store import (
        get_or_create_store,
        upload_file,
        query_store,
        extract_citations,
        delete_store,
    )

    logger.info("=" * 60)
    logger.info("TEST: Parse PDF -> Upload to Gemini -> Search")
    logger.info("=" * 60)

    store = None
    temp_md_path = None

    try:
        # 1. Parse the PDF document
        pdf_path = "docs/structured/crocodile.pdf"
        logger.info(f"Parsing document: {pdf_path}")

        markdown_content = parse_document(pdf_path)
        logger.info(f"Parsed {len(markdown_content)} characters")

        # 2. Save parsed content to temp file and upload to Gemini
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(markdown_content)
            temp_md_path = f.name

        store = get_or_create_store("crocodile-test-store")
        logger.info(f"Uploading to store: {store.name}")
        upload_file(store, temp_md_path)

        # 3. Search for keyword
        search_term = "035315147"
        logger.info(f"Searching for: {search_term}")

        response = query_store(store, f"Find any information about {search_term}")

        logger.info("--- Search Results ---")
        logger.info(f"Response: {response.text}")

        citations = extract_citations(response)
        if citations:
            logger.info("--- Citations ---")
            for c in citations:
                logger.info(f"  [{c['title']}]: {c['text_preview']}")

    finally:
        # 4. Cleanup
        if temp_md_path and os.path.exists(temp_md_path):
            os.unlink(temp_md_path)
            logger.info("Cleaned up temp file")

        if store:
            logger.info(f"Deleting test store: {store.name}")
            delete_store(store.name)
            logger.info("Store deleted")


if __name__ == "__main__":
    test_gemini_file_search()
