"""Token calculation utilities for agents.

This module provides shared token estimation and cost calculation
functions used by both SheetsAgent and DocumentAgent.
"""

from dataclasses import dataclass
from typing import Dict, Optional


# Pricing per million tokens (as of 2024)
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-5.1-codex-mini": {"input": 0.03, "output": 0.06},  # $0.03/1M in, $0.06/1M out
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Gemini models
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
}

# Default pricing for unknown models
DEFAULT_PRICING = {"input": 0.10, "output": 0.30}


@dataclass
class TokenEstimate:
    """Token usage estimate with cost calculation."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd
        }


def estimate_tokens(text: str, multiplier: float = 1.3) -> int:
    """Estimate token count from text.

    Uses word count with a multiplier to approximate tokens.
    The default multiplier of 1.3 works well for English text.

    Args:
        text: Input text to estimate tokens for
        multiplier: Token-to-word ratio (default 1.3)

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return int(len(text.split()) * multiplier)


def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a model.

    Args:
        model: Model name/ID

    Returns:
        Dict with 'input' and 'output' prices per million tokens
    """
    # Try exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Try partial match (for versioned models like gemini-2.5-flash-002)
    model_lower = model.lower()
    for known_model, pricing in MODEL_PRICING.items():
        if known_model.lower() in model_lower or model_lower in known_model.lower():
            return pricing

    return DEFAULT_PRICING


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: Optional[str] = None
) -> float:
    """Calculate estimated cost for token usage.

    Args:
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        model: Model name for pricing lookup (optional)

    Returns:
        Estimated cost in USD
    """
    pricing = get_model_pricing(model) if model else DEFAULT_PRICING

    # Pricing is per million tokens
    input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (completion_tokens * pricing["output"]) / 1_000_000

    return input_cost + output_cost


def calculate_token_usage(
    input_text: str,
    output_text: str,
    model: Optional[str] = None
) -> TokenEstimate:
    """Calculate token usage from input/output text.

    This is the main function to use for token estimation. It combines
    token counting and cost calculation into a single result.

    Args:
        input_text: The input/prompt text
        output_text: The output/completion text
        model: Model name for pricing lookup (optional)

    Returns:
        TokenEstimate with token counts and estimated cost
    """
    prompt_tokens = estimate_tokens(input_text)
    completion_tokens = estimate_tokens(output_text)
    total_tokens = prompt_tokens + completion_tokens

    estimated_cost = calculate_cost(prompt_tokens, completion_tokens, model)

    return TokenEstimate(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(estimated_cost, 8)
    )
