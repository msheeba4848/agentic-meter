"""Runaway agent demo: budget enforcement stops an infinite retry loop.

This is the headline use case. An agent in a bad retry loop would burn
unbounded money. agentledger raises BudgetExceeded so the loop fails
fast instead.

Run: python examples/runaway_agent.py
"""
from agentledger import Ledger, BudgetExceeded, get_current_ledger


def buggy_agent_loop():
    """A pretend agent that retries forever on the same prompt.
    Without a budget cap, this would run until your credit card cried.
    """
    ledger = get_current_ledger()
    attempts = 0
    while True:
        attempts += 1
        # Simulate an expensive call that keeps hitting max_tokens
        ledger.record(
            "openai",
            "gpt-4o",
            input_tokens=4000,
            output_tokens=1024,
            finish_reason="length",
            prompt_hash="stuck_on_this_prompt",
        )
        if attempts > 10_000:  # safety net for demo if budget is None
            break
    return "should never reach here"


if __name__ == "__main__":
    print("Running a buggy agent with a $0.50 hard cap...\n")
    try:
        with Ledger(budget="$0.50", name="runaway_demo") as ledger:
            buggy_agent_loop()
    except BudgetExceeded as e:
        print(f"Caught BudgetExceeded after {ledger.total_calls} calls")
        print(f"  Spent: ${e.spent:.4f}")
        print(f"  Cap:   ${e.budget:.4f}\n")

    print(ledger.summary())
