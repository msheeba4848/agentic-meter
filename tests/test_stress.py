"""Stress tests for waste decomposition and caching. No API calls needed."""
from agenticmeter import Ledger
import random

# ---- Test 1: waste decomposition under high volume ----
def test_waste_decomposition_at_scale():
    l = Ledger(name="scale_test")
    random.seed(42)

    # 1,000 calls with varied characteristics
    for i in range(1000):
        scenario = random.random()
        if scenario < 0.05:                                    # 5% fail
            l.record("openai", "gpt-4o", 0, 0, failed=True,
                     error=random.choice(["RateLimitError: x",
                                          "TimeoutError: x",
                                          "APIError: x"]))
        elif scenario < 0.20:                                  # 15% retries
            l.record("openai", "gpt-4o-mini", 800, 200,
                     prompt_hash=f"hash_{i % 50}",  # 50 unique prompts
                     finish_reason="stop")
        elif scenario < 0.30:                                  # 10% truncated
            l.record("openai", "gpt-4o", 1000, 1024,
                     finish_reason="length",
                     prompt_hash=f"long_{i}")
        else:                                                  # 70% normal
            l.record("openai", "gpt-4o-mini",
                     random.randint(100, 2000),
                     random.randint(50, 500),
                     finish_reason="stop")

    waste = l.wasted_cost_by_category()
    print(f"Calls: {l.total_calls}")
    print(f"Total spent: ${l.total_cost:.4f}")
    print(f"Waste by category: {waste}")
    print(f"Sum of categories: ${sum(waste.values()):.4f}")
    print(f"l.wasted_cost: ${l.wasted_cost:.4f}")

    # Invariants that should always hold
    assert abs(sum(waste.values()) - l.wasted_cost) < 1e-6, "category sum != wasted_cost"
    assert l.wasted_cost <= l.total_cost, "wasted > total"
    assert all(v >= 0 for v in waste.values()), "negative category"
    print("PASS: invariants hold under 1000-call load\n")


# ---- Test 2: missed caching opportunity detection ----
def test_missed_caching_detection():
    l = Ledger(name="cache_test")

    # 10 identical long prompts (should be flagged as missed opportunity)
    for _ in range(10):
        l.record("anthropic", "claude-3-5-sonnet-20241022",
                 input_tokens=3000, output_tokens=300,
                 prompt_hash="repeated_long")

    # 10 unique short prompts (should NOT be flagged)
    for i in range(10):
        l.record("openai", "gpt-4o-mini",
                 input_tokens=100, output_tokens=50,
                 prompt_hash=f"unique_{i}")

    # 3 already-cached calls (should NOT be flagged - already using caching)
    for _ in range(3):
        l.record("anthropic", "claude-3-5-sonnet-20241022",
                 input_tokens=500, output_tokens=200,
                 cached_read_tokens=2500,
                 prompt_hash="already_cached")

    missed = l.missed_caching_opportunities()
    print(f"Missed opportunities found: {len(missed)}")
    for m in missed:
        print(f"  {m['prompt_hash']} x{m['count']} could save ${m['estimated_savings']:.4f}")

    assert len(missed) == 1, f"expected 1 opportunity, got {len(missed)}"
    assert missed[0]["prompt_hash"] == "repeated_long"
    assert missed[0]["count"] == 10
    assert missed[0]["estimated_savings"] > 0
    print("PASS: correctly identified the one missed opportunity\n")


# ---- Test 3: cache write overhead (when caching is WORSE) ----
def test_cache_overhead_detection():
    """If a prompt is only called once, caching it would COST more (write surcharge)."""
    l = Ledger(name="overhead_test")

    # Cache-written but called only once — wasted the write overhead
    l.record("anthropic", "claude-3-5-sonnet-20241022",
             input_tokens=0, output_tokens=200,
             cached_write_tokens=2000, cached_read_tokens=0,
             prompt_hash="one_time")

    # Without caching, this would have been: 2000 * 3.00 / 1M = $0.0060
    # With caching write: 2000 * 3.75 / 1M = $0.0075 (25% MORE)
    print(f"Cost with cache write (one-time use): ${l.total_cost:.6f}")

    # Equivalent uncached
    l2 = Ledger()
    l2.record("anthropic", "claude-3-5-sonnet-20241022",
              input_tokens=2000, output_tokens=200)
    print(f"Cost without caching:                 ${l2.total_cost:.6f}")

    assert l.total_cost > l2.total_cost, "cache write should be more expensive than no-cache"
    print("PASS: cache write surcharge correctly applied (caching cost more)\n")


# ---- Test 4: pathological inputs ----
def test_edge_cases():
    l = Ledger()
    # Zero tokens
    c = l.record("openai", "gpt-4o", 0, 0)
    assert c.cost == 0.0

    # Negative tokens (shouldn't crash, even if nonsensical)
    c = l.record("openai", "gpt-4o", -5, -3)
    # Just verify it doesn't blow up

    # Unknown provider
    c = l.record("nonexistent_provider", "fake_model", 1000, 500)
    assert c.cost == 0.0  # unknown model -> 0 cost

    # Huge numbers
    c = l.record("openai", "gpt-4o", 10_000_000, 5_000_000)
    assert c.cost > 0
    assert c.cost < 1000  # sanity: shouldn't be billions of dollars

    print("PASS: edge cases handled\n")


# ---- Test 5: full summary doesn't crash on weird mixes ----
def test_summary_on_chaotic_data():
    l = Ledger(budget="$10.00", name="chaos")
    random.seed(0)

    for _ in range(200):
        l.record(
            random.choice(["openai", "anthropic"]),
            random.choice(["gpt-4o", "gpt-4o-mini",
                           "claude-3-5-sonnet-20241022",
                           "claude-3-5-haiku-20241022"]),
            random.randint(0, 5000),
            random.randint(0, 2000),
            cached_read_tokens=random.choice([0, 0, 0, 500, 1000]),
            cached_write_tokens=random.choice([0, 0, 0, 0, 1500]),
            finish_reason=random.choice(["stop", "length", "stop", "stop"]),
            failed=random.random() < 0.05,
            error="SomeError: msg" if random.random() < 0.05 else None,
            prompt_hash=f"h_{random.randint(0, 30)}",
        )

    summary = l.summary()
    assert len(summary) > 0
    print(summary[:2000])  # first 2000 chars
    print("\nPASS: summary renders on chaotic mixed data\n")


if __name__ == "__main__":
    test_waste_decomposition_at_scale()
    test_missed_caching_detection()
    test_cache_overhead_detection()
    test_edge_cases()
    test_summary_on_chaotic_data()
    print("=" * 50)
    print("All synthetic stress tests passed.")