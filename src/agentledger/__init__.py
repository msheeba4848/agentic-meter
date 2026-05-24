"""agentledger - FinOps for LLM agents.

Budget enforcement, counterfactual costs, waste tracking, and per-agent
attribution for OpenAI, Anthropic, and LangChain.

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

For LangChain (use the callback handler, not the raw-SDK monkey-patch):

    from agentledger import Ledger
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")
    with Ledger(budget="$1.00") as ledger:
        cb = ledger.as_langchain_callback()
        llm.invoke("hello", config={"callbacks": [cb]})
"""
from .ledger import Ledger, Call, track_budget, get_current_ledger
from .exceptions import BudgetExceeded, AgentLedgerError
from .pricing import calculate_cost, get_price, list_models, reload_prices

__version__ = "0.2.0"

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