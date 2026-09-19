"""
Embedding service for semantic caching using Gemini embeddings.

Generates query embeddings for semantic similarity matching in the RAG cache.

NOTE: these embeddings feed the semantic cache only - document retrieval is
handled by Gemini File Search and does not depend on this module. Callers should
treat embedding failures as non-fatal and fall back to an uncached query.
"""

import logging
import os
from typing import List, Optional

from google import genai
from google.genai import types

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


# Gemini embedding configuration.
#
# gemini-embedding-2 replaces text-embedding-004, which Google retired (the v1beta
# endpoint now returns 404 NOT_FOUND for it).
#
# The dimension MUST be requested explicitly: the model defaults to a larger width,
# while rag_query_cache.query_embedding is Vector(768) (see biz2bricks_core
# models/rag.py). At 768 this model returns unit-norm vectors, so no renormalisation
# is needed before the pgvector cosine comparison.
#
# Changing the model invalidates every stored embedding - vectors from different
# models are not comparable - so purge rag_query_cache when you change it.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))


async def get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a query using the configured Gemini embedding model.

    Args:
        query: The query text to embed

    Returns:
        List of floats representing the EMBEDDING_DIMENSION-wide embedding vector

    Raises:
        Exception: If embedding generation fails
    """
    try:
        client = get_client()

        # Generate embedding using Gemini
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            ),
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
        List of floats representing the EMBEDDING_DIMENSION-wide embedding vector
    """
    try:
        client = get_client()

        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIMENSION
            ),
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
