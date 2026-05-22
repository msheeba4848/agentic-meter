"""agentledger - FinOps for LLM agents.

Budget enforcement, counterfactual costs, and waste tracking for OpenAI and
Anthropic API calls.

Quickstart:

    from agentledger import Ledger
    import openai

    client = openai.OpenAI()
    with Ledger(budget="$2.00") as ledger:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    print(ledger.summary())
"""
from .ledger import Ledger, Call, track_budget, get_current_ledger
from .exceptions import BudgetExceeded, AgentLedgerError
from .pricing import calculate_cost, get_price, list_models, reload_prices

__version__ = "0.1.0"

__all__ = [
    "Ledger",
    "Call",
    "track_budget",
    "get_current_ledger",
    "BudgetExceeded",
    "AgentLedgerError",
    "calculate_cost",
    "get_price",
    "list_models",
    "reload_prices",
    "__version__",
]
