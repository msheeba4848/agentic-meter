"""Pricing data and cost calculations.

Prices are loaded from prices.json and cached. Keep this module dependency-free
so it works even without openai or anthropic installed.
"""
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

_PRICES_FILE = Path(__file__).parent / "prices.json"
_prices_cache: Optional[Dict] = None


def _load_prices() -> Dict:
    global _prices_cache
    if _prices_cache is None:
        with open(_PRICES_FILE) as f:
            data = json.load(f)
        # Drop the _meta key so it doesn't show up as a provider
        _prices_cache = {k: v for k, v in data.items() if not k.startswith("_")}
    return _prices_cache


def get_price(provider: str, model: str) -> Optional[Dict[str, float]]:
    """Get price per 1M tokens for a model.

    Returns dict with 'input' and 'output' keys, or None if model unknown.
    Handles dated model strings via prefix matching.
    """
    prices = _load_prices()
    provider_prices = prices.get(provider, {})
    if model in provider_prices:
        return provider_prices[model]
    # Prefix match: 'gpt-4o-2024-11-20' should match 'gpt-4o'
    candidates = [
        (known, price)
        for known, price in provider_prices.items()
        if model.startswith(known) or known.startswith(model)
    ]
    if candidates:
        # Prefer the longest known prefix
        candidates.sort(key=lambda x: -len(x[0]))
        return candidates[0][1]
    return None


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_read_tokens: int = 0,
    cached_write_tokens: int = 0,
) -> float:
    """Calculate cost in USD for a given call.

    ``input_tokens`` should be UNCACHED input tokens (billed at full rate).
    ``cached_read_tokens`` are input tokens served from a cache.
    ``cached_write_tokens`` are input tokens written to the cache (Anthropic-style).
    Returns 0.0 if the model is unknown.
    """
    price = get_price(provider, model)
    if price is None:
        return 0.0
    # Default cache rates fall back to the regular input rate if a model
    # doesn't define them - so behavior is unchanged for non-caching workloads.
    cached_read_rate = price.get("cached_read", price["input"])
    cached_write_rate = price.get("cached_write", price["input"])
    return (
        input_tokens * price["input"]
        + output_tokens * price["output"]
        + cached_read_tokens * cached_read_rate
        + cached_write_tokens * cached_write_rate
    ) / 1_000_000


def list_models(provider: Optional[str] = None) -> List[Tuple[str, str]]:
    """List all known (provider, model) pairs, optionally filtered by provider."""
    prices = _load_prices()
    if provider:
        return [(provider, m) for m in prices.get(provider, {}).keys()]
    return [(p, m) for p, models in prices.items() for m in models.keys()]


def reload_prices() -> None:
    """Force reload of prices from disk. Useful if prices.json is updated at runtime."""
    global _prices_cache
    _prices_cache = None
    _load_prices()