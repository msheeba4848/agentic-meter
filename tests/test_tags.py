"""Tests for the tag system (per-agent attribution)."""
from agenticmeter import Ledger


def test_basic_tag_attribution():
    ledger = Ledger()
    with ledger.tag("researcher"):
        ledger.record("openai", "gpt-4o-mini", 1000, 500)
    with ledger.tag("writer"):
        ledger.record("openai", "gpt-4o", 2000, 1000)

    by_tag = ledger.by_tag()
    assert "researcher" in by_tag
    assert "writer" in by_tag
    assert by_tag["researcher"] < by_tag["writer"]  # mini is cheaper
    # Sum equals total
    assert abs(sum(by_tag.values()) - ledger.total_cost) < 1e-9


def test_untagged_calls_grouped():
    ledger = Ledger()
    ledger.record("openai", "gpt-4o-mini", 1000, 500)  # no tag
    with ledger.tag("agent"):
        ledger.record("openai", "gpt-4o-mini", 1000, 500)

    by_tag = ledger.by_tag()
    assert "_untagged" in by_tag
    assert "agent" in by_tag


def test_nested_tags_attribute_to_innermost():
    ledger = Ledger()
    with ledger.tag("pipeline"):
        with ledger.tag("researcher"):
            ledger.record("openai", "gpt-4o-mini", 1000, 500)
        with ledger.tag("writer"):
            ledger.record("openai", "gpt-4o-mini", 1000, 500)

    by_tag = ledger.by_tag()
    # Innermost only - "pipeline" shouldn't appear
    assert "pipeline" not in by_tag
    assert "researcher" in by_tag
    assert "writer" in by_tag
    # No double counting
    assert abs(sum(by_tag.values()) - ledger.total_cost) < 1e-9


def test_tags_appear_on_call_object():
    ledger = Ledger()
    with ledger.tag("outer"):
        with ledger.tag("inner"):
            call = ledger.record("openai", "gpt-4o-mini", 100, 100)
    assert call.tags == ["outer", "inner"]


def test_tag_stack_cleared_on_exit():
    ledger = Ledger()
    with ledger.tag("temp"):
        pass
    # Stack should be empty after exit
    assert ledger._tag_stack == []
    # Call outside any tag should be _untagged
    ledger.record("openai", "gpt-4o-mini", 100, 100)
    assert ledger.by_tag().get("_untagged", 0) > 0


def test_calls_by_tag():
    ledger = Ledger()
    with ledger.tag("a"):
        ledger.record("openai", "gpt-4o-mini", 100, 100)
        ledger.record("openai", "gpt-4o-mini", 100, 100)
    with ledger.tag("b"):
        ledger.record("openai", "gpt-4o-mini", 100, 100)

    counts = ledger.calls_by_tag()
    assert counts["a"] == 2
    assert counts["b"] == 1


def test_summary_includes_by_tag_section():
    ledger = Ledger(name="test")
    with ledger.tag("researcher"):
        ledger.record("openai", "gpt-4o-mini", 1000, 500)
    with ledger.tag("writer"):
        ledger.record("openai", "gpt-4o", 1000, 500)

    text = ledger.summary()
    assert "By tag:" in text
    assert "researcher" in text
    assert "writer" in text


def test_summary_skips_by_tag_when_no_tags():
    ledger = Ledger()
    ledger.record("openai", "gpt-4o-mini", 1000, 500)
    text = ledger.summary()
    assert "By tag:" not in text