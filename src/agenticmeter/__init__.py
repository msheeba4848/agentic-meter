"""agenticmeter - FinOps for LLM agents.

Budget enforcement, counterfactual costs, waste tracking, caching analysis,
and per-agent attribution for OpenAI, Anthropic, AWS Bedrock, and LangChain.

Quickstart:

    from agenticmeter import Ledger
    import openai

    client = openai.OpenAI()
    with Ledger(budget="$2.00") as ledger:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    print(ledger.summary())

For AWS Bedrock (Claude, Llama, Titan) - any boto3 call is auto-tracked:

    import boto3
    from agenticmeter import Ledger

    bedrock = boto3.client("bedrock-runtime")
    with Ledger(budget="$1.00") as ledger:
        bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            body='{"messages":[...],"max_tokens":100,"anthropic_version":"bedrock-2023-05-31"}',
        )
    print(ledger.summary())

For custom wrappers (Azure, Vertex, in-house routers) - annotate once:

    from agenticmeter import track_llm_call, Ledger

    def _extract(response, args, kwargs):
        return {
            "provider": "anthropic",
            "model": kwargs["model"],
            "input_tokens": response["usage"]["input_tokens"],
            "output_tokens": response["usage"]["output_tokens"],
        }

    @track_llm_call(extract_usage=_extract)
    def my_wrapper(prompt, model, max_tokens):
        return call_my_provider(...)

For LangChain (use the callback handler, not the raw-SDK monkey-patch):

    from agenticmeter import Ledger
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")
    with Ledger(budget="$1.00") as ledger:
        cb = ledger.as_langchain_callback()
        llm.invoke("hello", config={"callbacks": [cb]})
"""
from .ledger import (
    Ledger,
    Call,
    track_budget,
    track_llm_call,
    tracked_executor,
    get_current_ledger,
)
from .exceptions import BudgetExceeded, AgenticMeterError, agenticmeterError
from .pricing import calculate_cost, get_price, list_models, reload_prices

__version__ = "0.4.0"

__all__ = [
    "Ledger",
    "Call",
    "track_budget",
    "track_llm_call",
    "tracked_executor",
    "get_current_ledger",
    "BudgetExceeded",
    "AgenticMeterError",
    "agenticmeterError",  # v0.3 back-compat
    "calculate_cost",
    "get_price",
    "list_models",
    "reload_prices",
    "__version__",
]
