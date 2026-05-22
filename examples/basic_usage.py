"""Basic usage of agentledger.

Run with: python examples/basic_usage.py

This example uses manual recording (no SDK calls) so it works without
API keys. See `examples/with_openai.py` for the real-SDK version.
"""
from agentledger import Ledger, track_budget


def manual_example():
    """Use Ledger as a context manager with manual recording."""
    print("=== Manual recording example ===\n")
    with Ledger(budget="$1.00", name="manual_demo") as ledger:
        ledger.record("openai", "gpt-4o-mini", 1500, 400, finish_reason="stop")
        ledger.record(
            "anthropic",
            "claude-3-5-sonnet-20241022",
            2000,
            800,
            finish_reason="stop",
        )
    print(ledger.summary())
    print()


@track_budget("$0.50", on_complete=lambda l: print(l.summary()))
def my_agent_run(query: str):
    """A decorated 'agent' function. The whole call runs inside a Ledger."""
    from agentledger import get_current_ledger
    # In real code these would be OpenAI/Anthropic SDK calls, auto-tracked.
    # Here we simulate via manual record() on the active ledger.
    ledger = get_current_ledger()
    ledger.record("openai", "gpt-4o-mini", 800, 200, finish_reason="stop")
    return f"Answer to: {query}"


def decorator_example():
    print("=== Decorator example ===\n")
    result = my_agent_run("What is 2+2?")
    print(f"\nResult: {result}\n")


if __name__ == "__main__":
    manual_example()
    decorator_example()
