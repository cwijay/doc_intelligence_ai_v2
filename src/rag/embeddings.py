"""
Embedding service for semantic caching using Gemini text-embedding-004.

Generates query embeddings for semantic similarity matching in the RAG cache.
"""

import logging
import os
from typing import List, Optional

from google import genai

logger = logging.getLogger(__name__)

# Initialize Gemini client
_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    """Get or create the Gemini client."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            _client = genai.Client(api_key=api_key)
        else:
            _client = genai.Client()  # Use ADC
    return _client


# Gemini text-embedding-004 configuration
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 768  # Output dimension for text-embedding-004


async def get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a query using Gemini text-embedding-004.

    Args:
        query: The query text to embed

    Returns:
        List of floats representing the 768-dimensional embedding vector

    Raises:
        Exception: If embedding generation fails
    """
    try:
        client = get_client()

        # Generate embedding using Gemini
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )

        # Extract embedding values
        embedding = result.embeddings[0].values

        logger.debug(f"Generated embedding for query: '{query[:50]}...' (dim={len(embedding)})")

        return list(embedding)

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


def get_query_embedding_sync(query: str) -> List[float]:
    """
    Synchronous version of get_query_embedding for use in non-async contexts.

    Args:
        query: The query text to embed

    Returns:
        List of floats representing the 768-dimensional embedding vector
    """
    try:
        client = get_client()

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )

        embedding = result.embeddings[0].values
        logger.debug(f"Generated embedding for query: '{query[:50]}...' (dim={len(embedding)})")

        return list(embedding)

    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        raise


async def get_batch_embeddings(queries: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple queries in batch.

    Args:
        queries: List of query texts to embed

    Returns:
        List of embedding vectors, one per query
    """
    embeddings = []
    for query in queries:
        embedding = await get_query_embedding(query)
        embeddings.append(embedding)
    return embeddings
