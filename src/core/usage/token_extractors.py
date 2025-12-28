"""
Provider-specific token extraction from LLM responses.

Extracts actual token counts from OpenAI and Gemini response metadata
instead of word-count estimation.
"""

import logging
from typing import Any, Optional

from .schemas import TokenUsage

logger = logging.getLogger(__name__)


def extract_openai_tokens(response: Any) -> Optional[TokenUsage]:
    """
    Extract tokens from OpenAI response.

    OpenAI response structure (with responses API):
    - response.usage.prompt_tokens
    - response.usage.completion_tokens
    - response.usage.total_tokens
    - response.usage.prompt_tokens_details.cached_tokens (optional)

    Args:
        response: OpenAI LLM response object

    Returns:
        TokenUsage if extraction successful, None otherwise
    """
    try:
        # Try to get usage from response object
        usage = getattr(response, 'usage', None)
        if not usage:
            # Try dict access for dict-like responses
            if isinstance(response, dict):
                usage = response.get('usage')
            if not usage:
                logger.debug("No usage data found in OpenAI response")
                return None

        # Extract token counts
        if hasattr(usage, 'prompt_tokens'):
            # Object-style access
            input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
            output_tokens = getattr(usage, 'completion_tokens', 0) or 0
            total_tokens = getattr(usage, 'total_tokens', 0) or (input_tokens + output_tokens)
        else:
            # Dict-style access
            input_tokens = usage.get('prompt_tokens', 0) or 0
            output_tokens = usage.get('completion_tokens', 0) or 0
            total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)

        # Extract cached tokens (prompt caching feature)
        cached_tokens = 0
        prompt_details = getattr(usage, 'prompt_tokens_details', None)
        if prompt_details:
            if hasattr(prompt_details, 'cached_tokens'):
                cached_tokens = getattr(prompt_details, 'cached_tokens', 0) or 0
            elif isinstance(prompt_details, dict):
                cached_tokens = prompt_details.get('cached_tokens', 0) or 0

        # Get model name
        model = getattr(response, 'model', None)
        if not model and isinstance(response, dict):
            model = response.get('model')

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            provider="openai",
            model=model or "unknown",
        )

    except Exception as e:
        logger.warning(f"Failed to extract OpenAI tokens: {e}")
        return None


def extract_gemini_tokens(response: Any) -> Optional[TokenUsage]:
    """
    Extract tokens from Gemini response.

    Gemini response structure:
    - response.usage_metadata.prompt_token_count
    - response.usage_metadata.candidates_token_count
    - response.usage_metadata.total_token_count
    - response.usage_metadata.cached_content_token_count (optional)

    Args:
        response: Gemini LLM response object

    Returns:
        TokenUsage if extraction successful, None otherwise
    """
    try:
        # Try to get usage_metadata from response
        usage = getattr(response, 'usage_metadata', None)
        if not usage:
            # Try dict access
            if isinstance(response, dict):
                usage = response.get('usage_metadata') or response.get('usageMetadata')
            if not usage:
                logger.debug("No usage_metadata found in Gemini response")
                return None

        # Extract token counts
        if hasattr(usage, 'prompt_token_count'):
            # Object-style access
            input_tokens = getattr(usage, 'prompt_token_count', 0) or 0
            output_tokens = getattr(usage, 'candidates_token_count', 0) or 0
            total_tokens = getattr(usage, 'total_token_count', 0) or (input_tokens + output_tokens)
            cached_tokens = getattr(usage, 'cached_content_token_count', 0) or 0
        else:
            # Dict-style access (camelCase from API)
            input_tokens = usage.get('promptTokenCount', usage.get('prompt_token_count', 0)) or 0
            output_tokens = usage.get('candidatesTokenCount', usage.get('candidates_token_count', 0)) or 0
            total_tokens = usage.get('totalTokenCount', usage.get('total_token_count', 0)) or (input_tokens + output_tokens)
            cached_tokens = usage.get('cachedContentTokenCount', usage.get('cached_content_token_count', 0)) or 0

        # Get model name
        model = getattr(response, 'model', None)
        if not model and isinstance(response, dict):
            model = response.get('model')

        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            provider="google",
            model=model or "unknown",
        )

    except Exception as e:
        logger.warning(f"Failed to extract Gemini tokens: {e}")
        return None


def extract_tokens(response: Any, provider: Optional[str] = None) -> TokenUsage:
    """
    Extract tokens from LLM response, auto-detecting provider if needed.

    Args:
        response: LLM response object
        provider: Optional provider hint ('openai', 'google')

    Returns:
        TokenUsage (with zero values if extraction fails)
    """
    usage = None

    # Try provider-specific extraction
    if provider == "openai":
        usage = extract_openai_tokens(response)
    elif provider in ("google", "google_genai"):
        usage = extract_gemini_tokens(response)
    else:
        # Auto-detect provider
        # Check for OpenAI-style response
        if hasattr(response, 'usage') or (isinstance(response, dict) and 'usage' in response):
            usage = extract_openai_tokens(response)

        # Check for Gemini-style response
        if not usage:
            if hasattr(response, 'usage_metadata') or (isinstance(response, dict) and ('usage_metadata' in response or 'usageMetadata' in response)):
                usage = extract_gemini_tokens(response)

    # Return extracted usage or empty
    if usage:
        return usage

    # Fallback: try to estimate from content
    return _estimate_from_content(response)


def extract_from_langchain_response(response: Any) -> TokenUsage:
    """
    Extract tokens from LangChain LLMResult.

    LangChain stores token info in response.llm_output['token_usage'].

    Args:
        response: LangChain LLMResult object

    Returns:
        TokenUsage
    """
    try:
        # Get llm_output
        llm_output = getattr(response, 'llm_output', None)
        if not llm_output:
            return TokenUsage()

        # Check for token_usage in llm_output
        token_usage = llm_output.get('token_usage')
        if token_usage:
            return TokenUsage(
                input_tokens=token_usage.get('prompt_tokens', 0),
                output_tokens=token_usage.get('completion_tokens', 0),
                total_tokens=token_usage.get('total_tokens', 0),
                cached_tokens=token_usage.get('cached_tokens', 0),
                provider=llm_output.get('model_provider'),
                model=llm_output.get('model_name'),
            )

        # Check for usage_metadata (Gemini via LangChain)
        usage_metadata = llm_output.get('usage_metadata')
        if usage_metadata:
            return TokenUsage(
                input_tokens=usage_metadata.get('prompt_token_count', 0),
                output_tokens=usage_metadata.get('candidates_token_count', 0),
                total_tokens=usage_metadata.get('total_token_count', 0),
                cached_tokens=usage_metadata.get('cached_content_token_count', 0),
                provider="google",
                model=llm_output.get('model_name'),
            )

        return TokenUsage()

    except Exception as e:
        logger.warning(f"Failed to extract tokens from LangChain response: {e}")
        return TokenUsage()


def _estimate_from_content(response: Any) -> TokenUsage:
    """
    Fallback: estimate tokens from content length.

    Uses word count * 1.3 multiplier as rough estimate.
    This is less accurate than actual token counts.

    Args:
        response: Any response object

    Returns:
        TokenUsage with estimated values
    """
    try:
        # Try to get content from response
        content = ""

        if hasattr(response, 'content'):
            content = str(response.content)
        elif hasattr(response, 'text'):
            content = str(response.text)
        elif isinstance(response, str):
            content = response
        elif isinstance(response, dict):
            content = str(response.get('content', response.get('text', '')))

        if not content:
            return TokenUsage()

        # Estimate: ~1.3 tokens per word
        word_count = len(content.split())
        estimated_tokens = int(word_count * 1.3)

        logger.debug(f"Estimated {estimated_tokens} tokens from {word_count} words (fallback)")

        return TokenUsage(
            input_tokens=0,  # Can't estimate input from output
            output_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
        )

    except Exception as e:
        logger.warning(f"Failed to estimate tokens from content: {e}")
        return TokenUsage()


__all__ = [
    "extract_openai_tokens",
    "extract_gemini_tokens",
    "extract_tokens",
    "extract_from_langchain_response",
]
