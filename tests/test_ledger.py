"""Tests for the Ledger class. These run without any provider SDK installed."""
import pytest
from agentledger import Ledger, BudgetExceeded, get_current_ledger


def test_basic_recording():
    ledger = Ledger()
    ledger.record("openai", "gpt-4o-mini", input_tokens=1000, output_tokens=500)
    assert ledger.total_calls == 1
    # 1000 * 0.15/1M + 500 * 0.60/1M = 0.00015 + 0.0003 = 0.00045
    assert abs(ledger.total_cost - 0.00045) < 1e-9


def test_context_manager_sets_current_ledger():
    assert get_current_ledger() is None
    with Ledger() as ledger:
        assert get_current_ledger() is ledger
    assert get_current_ledger() is None


def test_budget_parsing_accepts_dollar_strings():
    l1 = Ledger(budget="$2.00")
    l2 = Ledger(budget="2.00")
    l3 = Ledger(budget=2.0)
    assert l1.budget == l2.budget == l3.budget == 2.0


def test_budget_enforcement_raises():
    ledger = Ledger(budget="$0.0001")
    with pytest.raises(BudgetExceeded) as exc_info:
        # 1M input tokens on gpt-4o-mini = $0.15, way over $0.0001
        ledger.record("openai", "gpt-4o-mini", 1_000_000, 0)
    assert exc_info.value.budget == 0.0001
    assert exc_info.value.spent > 0.0001


def test_budget_callback_fires_before_exception():
    callback_called = []
    ledger = Ledger(
        budget="$0.0001",
        on_budget_exceeded=lambda l: callback_called.append(l.total_cost),
    )
    with pytest.raises(BudgetExceeded):
        ledger.record("openai", "gpt-4o-mini", 1_000_000, 0)
    assert len(callback_called) == 1
    assert callback_called[0] > 0.0001


def test_no_budget_means_no_enforcement():
    ledger = Ledger()  # no budget
    # Should not raise no matter how much we record
    for _ in range(5):
        ledger.record("openai", "gpt-4o", 1_000_000, 1_000_000)
    assert ledger.total_cost > 0


def test_remaining_budget():
    ledger = Ledger(budget="$1.00")
    ledger.record("openai", "gpt-4o-mini", 1_000_000, 0)  # $0.15
    assert abs(ledger.remaining_budget - 0.85) < 1e-9


def test_counterfactual_cost():
    ledger = Ledger()
    # Record a gpt-4o call: 1M in, 1M out = $2.50 + $10 = $12.50
    ledger.record("openai", "gpt-4o", 1_000_000, 1_000_000)
    assert abs(ledger.total_cost - 12.50) < 1e-6
    # Same workload on gpt-4o-mini = $0.15 + $0.60 = $0.75
    cf = ledger.counterfactual_cost("openai", "gpt-4o-mini")
    assert abs(cf - 0.75) < 1e-6


def test_cheapest_alternative():
    ledger = Ledger()
    ledger.record("openai", "gpt-4o", 1_000, 1_000)
    result = ledger.cheapest_alternative()
    assert result is not None
    provider, model, cost = result
    # claude-3-haiku is the cheapest in our seed prices
    assert cost > 0


def test_failed_calls_excluded_from_total():
    ledger = Ledger()
    ledger.record("openai", "gpt-4o", 1000, 1000)  # succeeds
    ledger.record("openai", "gpt-4o", 0, 0, failed=True, error="boom")
    assert ledger.successful_calls == 1
    assert ledger.failed_calls == 1
    # Failed call contributes $0 to total
    assert ledger.total_cost > 0


def test_summary_renders_without_errors():
    ledger = Ledger(budget="$1.00", name="test_run")
    ledger.record("openai", "gpt-4o-mini", 1000, 500)
    ledger.record("anthropic", "claude-3-5-haiku-20241022", 2000, 1000)
    text = ledger.summary()
    assert "test_run" in text
    assert "openai" in text
    assert "anthropic" in text
    assert "$" in text


def test_to_dict_and_to_json():
    ledger = Ledger(budget="$1.00")
    ledger.record("openai", "gpt-4o-mini", 1000, 500, finish_reason="stop")
    d = ledger.to_dict()
    assert d["total_calls"] == 1
    assert d["budget"] == 1.0
    assert len(d["calls"]) == 1
    # Should be JSON-serializable
    import json
    json.loads(ledger.to_json())
