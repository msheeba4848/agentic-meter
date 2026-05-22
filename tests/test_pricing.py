"""Tests for the pricing module."""
from agentledger.pricing import (
    get_price,
    calculate_cost,
    list_models,
)


def test_get_price_known_openai_model():
    price = get_price("openai", "gpt-4o-mini")
    assert price is not None
    assert price["input"] == 0.15
    assert price["output"] == 0.60


def test_get_price_known_anthropic_model():
    price = get_price("anthropic", "claude-3-5-haiku-20241022")
    assert price is not None
    assert price["input"] == 0.80
    assert price["output"] == 4.00


def test_get_price_unknown_model_returns_none():
    assert get_price("openai", "gpt-99-impossibilon") is None
    assert get_price("nonexistent_provider", "anything") is None


def test_get_price_prefix_match():
    # Future-dated model strings should still resolve via prefix match
    price = get_price("openai", "gpt-4o-mini-2099-01-01")
    assert price is not None
    assert price["input"] == 0.15


def test_calculate_cost_basic():
    # 1M input + 1M output on gpt-4o-mini = 0.15 + 0.60 = 0.75
    cost = calculate_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert abs(cost - 0.75) < 1e-9


def test_calculate_cost_partial_tokens():
    # 1000 input tokens on gpt-4o ($2.50/1M) = $0.0025
    cost = calculate_cost("openai", "gpt-4o", 1000, 0)
    assert abs(cost - 0.0025) < 1e-9


def test_calculate_cost_unknown_model_is_zero():
    cost = calculate_cost("openai", "made-up-model", 1000, 1000)
    assert cost == 0.0


def test_list_models_filtered_by_provider():
    openai_models = list_models("openai")
    assert all(p == "openai" for p, _ in openai_models)
    assert len(openai_models) > 0
    assert ("openai", "gpt-4o-mini") in openai_models


def test_list_models_all():
    all_models = list_models()
    providers = {p for p, _ in all_models}
    assert "openai" in providers
    assert "anthropic" in providers
