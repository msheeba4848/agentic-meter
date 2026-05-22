"""Tests for waste detection: retries, truncations, failed calls."""
import time
from agentledger import Ledger


def test_retry_detection_same_prompt_hash():
    ledger = Ledger()
    h = "same_hash_xyz"
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash=h)
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash=h)
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash=h)
    retries = ledger.detect_retries()
    # First call is the original; 2nd and 3rd are retries
    assert len(retries) == 2


def test_no_retries_with_different_hashes():
    ledger = Ledger()
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash="a")
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash="b")
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash="c")
    assert ledger.detect_retries() == []


def test_retry_window_respected():
    ledger = Ledger(retry_window_seconds=0.001)  # tiny window
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash="same")
    time.sleep(0.01)  # exceed the window
    ledger.record("openai", "gpt-4o-mini", 100, 100, prompt_hash="same")
    # Second call is outside the retry window, so not flagged
    assert ledger.detect_retries() == []


def test_truncation_detection_openai_style():
    ledger = Ledger()
    ledger.record(
        "openai", "gpt-4o-mini", 100, 500, finish_reason="length"
    )
    ledger.record(
        "openai", "gpt-4o-mini", 100, 100, finish_reason="stop"
    )
    truncs = ledger.detect_truncations()
    assert len(truncs) == 1
    assert truncs[0].finish_reason == "length"


def test_truncation_detection_anthropic_style():
    ledger = Ledger()
    ledger.record(
        "anthropic",
        "claude-3-5-haiku-20241022",
        100,
        500,
        finish_reason="max_tokens",
    )
    truncs = ledger.detect_truncations()
    assert len(truncs) == 1


def test_failed_calls_counted_as_waste():
    ledger = Ledger()
    ledger.record("openai", "gpt-4o", 1000, 1000)  # success
    ledger.record(
        "openai", "gpt-4o", 0, 0, failed=True, error="ratelimit"
    )
    failed = ledger.detect_failed()
    assert len(failed) == 1


def test_wasted_cost_aggregates_across_categories():
    ledger = Ledger()
    # A normal call
    ledger.record("openai", "gpt-4o-mini", 1000, 1000)
    # A truncated call (still counts as cost)
    ledger.record(
        "openai", "gpt-4o-mini", 1000, 1000, finish_reason="length"
    )
    # A retry of the truncated call
    ledger.record(
        "openai",
        "gpt-4o-mini",
        1000,
        1000,
        finish_reason="length",
        prompt_hash="dupe",
    )
    ledger.record(
        "openai",
        "gpt-4o-mini",
        1000,
        1000,
        finish_reason="length",
        prompt_hash="dupe",
    )
    # Wasted cost > 0 and less than or equal to total
    assert ledger.wasted_cost > 0
    assert ledger.wasted_cost <= ledger.total_cost
