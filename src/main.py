"""
FastAPI service for Document Intelligence AI v3.0.

This service provides a comprehensive REST API for:
- Document Agent: Summaries, FAQs, questions generation
- Sheets Agent: Excel/CSV analysis with natural language
- Document ingestion and parsing (LlamaParse)
- Semantic search via Gemini File Store
- Full audit trail and analytics

Usage:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8001
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to Python path for direct execution
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables before importing anything else
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import and create the FastAPI app
from src.api import create_app

app = create_app()

logger.info("Document Intelligence AI v3.0 initialized")
logger.info("API documentation available at /docs and /redoc")


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8001"))
    reload = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "src.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )
