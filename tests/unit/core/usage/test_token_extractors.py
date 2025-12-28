"""Tests for token extraction from LLM responses."""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_core.outputs import LLMResult, Generation

from src.core.usage.token_extractors import (
    extract_openai_tokens,
    extract_gemini_tokens,
    extract_from_langchain_response,
    extract_tokens,
)
from src.core.usage.schemas import TokenUsage


class TestExtractOpenAITokens:
    """Tests for OpenAI token extraction."""

    def test_extract_with_usage_metadata(self):
        """Test extraction when response has usage metadata."""
        mock_response = Mock(spec=[])
        mock_response.usage = Mock(spec=[])
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details = None  # Explicitly set to None
        mock_response.model = "gpt-4o"

        result = extract_openai_tokens(mock_response)

        assert result is not None
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.provider == "openai"
        assert result.model == "gpt-4o"

    def test_extract_without_usage_metadata(self):
        """Test extraction returns None when no usage metadata."""
        mock_response = Mock(spec=[])
        mock_response.usage = None

        result = extract_openai_tokens(mock_response)

        assert result is None

    def test_extract_with_cached_tokens(self):
        """Test extraction includes cached tokens when available."""
        mock_response = Mock(spec=[])
        mock_response.usage = Mock(spec=[])
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details = {"cached_tokens": 20}
        mock_response.model = "gpt-4o"

        result = extract_openai_tokens(mock_response)

        assert result is not None
        assert result.cached_tokens == 20


class TestExtractGeminiTokens:
    """Tests for Gemini token extraction."""

    def test_extract_with_usage_metadata(self):
        """Test extraction when response has usage_metadata."""
        mock_response = Mock(spec=[])
        mock_response.usage_metadata = Mock(spec=[])
        mock_response.usage_metadata.prompt_token_count = 200
        mock_response.usage_metadata.candidates_token_count = 100
        mock_response.usage_metadata.total_token_count = 300
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.model = "gemini-3-flash-preview"

        result = extract_gemini_tokens(mock_response)

        assert result is not None
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.total_tokens == 300
        assert result.provider == "google"
        assert result.model == "gemini-3-flash-preview"

    def test_extract_without_usage_metadata(self):
        """Test extraction returns None when no usage_metadata."""
        mock_response = Mock(spec=[])
        mock_response.usage_metadata = None

        result = extract_gemini_tokens(mock_response)

        assert result is None

    def test_extract_with_cached_content_tokens(self):
        """Test extraction includes cached tokens when available."""
        mock_response = Mock(spec=[])
        mock_response.usage_metadata = Mock(spec=[])
        mock_response.usage_metadata.prompt_token_count = 200
        mock_response.usage_metadata.candidates_token_count = 100
        mock_response.usage_metadata.total_token_count = 300
        mock_response.usage_metadata.cached_content_token_count = 50
        mock_response.model = "gemini-3-flash-preview"

        result = extract_gemini_tokens(mock_response)

        assert result is not None
        assert result.cached_tokens == 50


class TestExtractFromLangchainResponse:
    """Tests for LangChain LLMResult extraction."""

    def test_extract_from_llm_output_token_usage(self):
        """Test extraction from llm_output token_usage dict."""
        mock_result = LLMResult(
            generations=[[Generation(text="Hello")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "model_name": "gpt-4o",
            },
        )

        result = extract_from_langchain_response(mock_result)

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        assert result.model == "gpt-4o"

    def test_extract_from_llm_output_usage_metadata(self):
        """Test extraction from llm_output usage_metadata dict (Gemini style)."""
        mock_result = LLMResult(
            generations=[[Generation(text="Hello")]],
            llm_output={
                "usage_metadata": {
                    "prompt_token_count": 200,
                    "candidates_token_count": 100,
                    "total_token_count": 300,
                },
                "model_name": "gemini-3-flash-preview",
            },
        )

        result = extract_from_langchain_response(mock_result)

        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.total_tokens == 300

    def test_extract_without_llm_output_returns_empty(self):
        """Test that missing llm_output returns empty TokenUsage."""
        mock_result = LLMResult(
            generations=[[Generation(text="Hello world")]],
            llm_output=None,
        )

        result = extract_from_langchain_response(mock_result)

        # Returns empty TokenUsage when llm_output is None
        assert result.total_tokens == 0

    def test_extract_empty_generations_returns_zeros(self):
        """Test extraction with empty generations."""
        mock_result = LLMResult(
            generations=[],
            llm_output=None,
        )

        result = extract_from_langchain_response(mock_result)

        assert result.total_tokens == 0


class TestExtractTokens:
    """Tests for extract_tokens auto-detection."""

    def test_extract_tokens_openai_with_provider_hint(self):
        """Test extraction with OpenAI provider hint."""
        mock_response = Mock(spec=[])
        mock_response.usage = Mock(spec=[])
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details = None
        mock_response.model = "gpt-4o"

        result = extract_tokens(mock_response, provider="openai")

        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.provider == "openai"

    def test_extract_tokens_gemini_with_provider_hint(self):
        """Test extraction with Gemini provider hint."""
        mock_response = Mock(spec=[])
        mock_response.usage_metadata = Mock(spec=[])
        mock_response.usage_metadata.prompt_token_count = 200
        mock_response.usage_metadata.candidates_token_count = 100
        mock_response.usage_metadata.total_token_count = 300
        mock_response.usage_metadata.cached_content_token_count = 0
        mock_response.model = "gemini-3-flash-preview"

        result = extract_tokens(mock_response, provider="google")

        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.provider == "google"

    def test_extract_tokens_auto_detect_openai(self):
        """Test auto-detection of OpenAI response."""
        mock_response = Mock(spec=[])
        mock_response.usage = Mock(spec=[])
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_response.usage.prompt_tokens_details = None
        mock_response.model = "gpt-4o"

        result = extract_tokens(mock_response)

        assert result.input_tokens == 100
        assert result.provider == "openai"

    def test_extract_tokens_fallback_returns_empty(self):
        """Test that failed extraction returns empty or estimated TokenUsage."""
        mock_response = Mock(spec=[])
        mock_response.usage = None
        mock_response.usage_metadata = None
        mock_response.content = ""  # Empty string, no content to estimate
        mock_response.text = ""

        result = extract_tokens(mock_response)

        # With empty content, should return 0 or small estimated value
        assert result.total_tokens >= 0
