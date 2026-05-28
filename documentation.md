# agenticmeter Documentation

In-depth reference for the agenticmeter package. For the elevator pitch and install instructions, see [`README.md`](README.md). For the version history, see [`CHANGELOG.md`](CHANGELOG.md).

---

## Table of contents

1. [Installation](#installation)
2. [Core concepts](#core-concepts)
3. [The Ledger class](#the-ledger-class)
4. [Integration guides](#integration-guides)
5. [Budget enforcement](#budget-enforcement)
6. [Per-agent attribution (tags)](#per-agent-attribution-tags)
7. [Waste decomposition](#waste-decomposition)
8. [Caching analysis](#caching-analysis)
9. [Counterfactual costs](#counterfactual-costs)
10. [Agent-loop detection](#agent-loop-detection)
11. [The LangChain callback](#the-langchain-callback)
12. [Manual recording](#manual-recording)
13. [Pricing data](#pricing-data)
14. [Programmatic API reference](#programmatic-api-reference)
15. [Working with threads and async](#working-with-threads-and-async)
16. [Troubleshooting](#troubleshooting)

---

## Installation

Requires Python 3.9 or later.

```bash
pip install agenticmeter              # core only, zero runtime dependencies
pip install agenticmeter[openai]      # adds openai>=1.0.0
pip install agenticmeter[anthropic]   # adds anthropic>=0.20.0
pip install agenticmeter[langchain]   # adds langchain-core>=0.3.0
pip install agenticmeter[all]         # everything above
pip install agenticmeter[dev]         # for contributors: adds pytest, pytest-asyncio
```

For local development:

```bash
git clone https://github.com/msheeba4848/agentic-meter.git
cd agenticmeter
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/
```

---

## Core concepts

**Ledger.** The unit of tracking. A `Ledger` is a Python context manager that captures every LLM call made inside its `with` block. One ledger = one logical "run" — a script, an agent invocation, a user conversation.

**Call.** A single recorded LLM call. Has provider, model, input/output tokens, cost in USD, duration, finish reason, and optional metadata. Calls are stored on the Ledger in the order they happened.

**Tag.** A label attached to a group of calls via `ledger.tag("name")`. Used to attribute cost to specific agents, features, or pipeline phases. Tags nest; the innermost tag is used for primary attribution.

**Budget.** An optional USD cap on the ledger. When exceeded, the *next* recorded call raises `BudgetExceeded`. The exception fires after the call that crossed the cap completes — so the budget is approximate to within one call's cost.

**Active ledger.** The ledger currently inside a `with` block. Stored in a `contextvars.ContextVar` so it propagates correctly across async tasks. The SDK monkey-patches and the LangChain callback both read this to know where to record.

---

## The Ledger class

```python
from agenticmeter import Ledger

ledger = Ledger(
    budget="$5.00",                # cap in USD; raises BudgetExceeded when exceeded
    name="my_workflow",            # optional label, shown in summary
    retry_window_seconds=60.0,     # window for retry detection
    on_budget_exceeded=None,       # optional callback(ledger) before raise
)
```

### As a context manager

```python
with Ledger(budget="$1.00") as ledger:
    # Any OpenAI or Anthropic SDK call inside is auto-tracked
    response = openai_client.chat.completions.create(...)
print(ledger.summary())
```

When the `with` block exits, the SDK patches stop recording (they check whether a Ledger is currently active via `contextvars`). The Ledger object itself remains valid — you can call `summary()`, `to_dict()`, `to_json()`, and other methods after the block exits.

### Using the decorator form

```python
from agenticmeter import track_budget

@track_budget("$0.50", name="qa_agent")
def answer_question(q):
    return llm.invoke(q)
```

### Accessing the active ledger from inside any function

```python
from agenticmeter import get_current_ledger

def my_function():
    ledger = get_current_ledger()
    if ledger is not None:
        ledger.record(provider="custom", model="my-model", input_tokens=100, output_tokens=50)
```

---

## Integration guides

### OpenAI

```python
from agenticmeter import Ledger
from openai import OpenAI, AsyncOpenAI

# Sync
client = OpenAI()
with Ledger(budget="$1.00") as ledger:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

# Async
async_client = AsyncOpenAI()
async def run():
    with Ledger(budget="$1.00") as ledger:
        response = await async_client.chat.completions.create(...)
```

The patch covers `Completions.create` and `AsyncCompletions.create`. Streaming (`stream=True`) is not yet tracked — call won't crash but tokens will show as 0. Use non-streaming for accurate accounting until v0.4.

### Anthropic

```python
from agenticmeter import Ledger
from anthropic import Anthropic, AsyncAnthropic

client = Anthropic()
with Ledger(budget="$1.00") as ledger:
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "hi"}],
    )
```

Same caveats as OpenAI re streaming.

### LangChain

LangChain wraps responses differently than the raw SDKs, so we provide a callback handler that extracts tokens correctly. See [The LangChain callback](#the-langchain-callback) for the full guide.

```python
from agenticmeter import Ledger
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
with Ledger(budget="$1.00") as ledger:
    cb = ledger.as_langchain_callback()
    response = llm.invoke("hi", config={"callbacks": [cb]})
```

### Other providers (Bedrock, Azure, Vertex, custom)

Native auto-tracking for these is planned for v0.4. For now use manual recording:

```python
with Ledger() as ledger:
    response = bedrock_client.invoke_model(...)
    # Extract from your response shape
    ledger.record(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        input_tokens=response["usage"]["input_tokens"],
        output_tokens=response["usage"]["output_tokens"],
    )
```

If you have a wrapper function that every call passes through, instrument that function once and you're done.

---

## Budget enforcement

The killer feature. Set a cap; runaway agents fail fast.

```python
from agenticmeter import Ledger, BudgetExceeded

try:
    with Ledger(budget="$5.00") as ledger:
        my_buggy_agent()
except BudgetExceeded as e:
    log.error(f"Agent stopped at ${e.spent:.4f} (cap ${e.budget:.4f})")
    log.info(ledger.summary())
```

The exception fires on the *next call after* the cap is crossed. The call that breaks the cap completes — it can't be cancelled in flight. So if your cap is $0.50 and a single call costs $0.60, you'll spend $0.60 before the budget catches up. For sane budgets relative to per-call cost, this overshoot is negligible.

### Budget formats

```python
Ledger(budget="$2.00")    # dollar-prefixed string
Ledger(budget="2.00")     # plain string
Ledger(budget=2.0)        # float
Ledger(budget=None)       # no enforcement (default)
```

### Alert callback before raise

For sending Slack/email alerts before the exception fires:

```python
def alert(ledger):
    slack.post(f"agent over budget: ${ledger.total_cost:.2f}")

with Ledger(budget="$10.00", on_budget_exceeded=alert) as ledger:
    run_agent()
```

The callback is called, then `BudgetExceeded` is raised. If the callback itself raises, the exception is swallowed (you don't want a logging failure to mask a budget violation).

---

## Per-agent attribution (tags)

Tags break down cost by feature, agent, or pipeline phase.

```python
with Ledger(budget="$5.00") as ledger:
    with ledger.tag("researcher"):
        research = research_llm.invoke("research topic")
    with ledger.tag("writer"):
        draft = writer_llm.invoke(f"write about: {research}")
    with ledger.tag("critic"):
        review = critic_llm.invoke(f"critique: {draft}")

print(ledger.by_tag())
# {'researcher': 0.0014, 'writer': 0.0058, 'critic': 0.0023}
```

### Nested tags

Tags compose. The innermost tag is used for primary attribution in `by_tag()`:

```python
with ledger.tag("pipeline"):
    with ledger.tag("researcher"):
        # This call's tags are ["pipeline", "researcher"]
        # by_tag() attributes it to "researcher"
        llm.invoke(...)
```

The outer tag is preserved on the `Call.tags` list — useful if you want to slice differently.

### Tags from outside `with` blocks

```python
from contextlib import nullcontext
from agenticmeter import get_current_ledger

def _tag(name):
    ledger = get_current_ledger()
    return ledger.tag(name) if ledger else nullcontext()

# Now any function can tag without checking for an active ledger
def my_agent_step():
    with _tag("step_a"):
        ...
```

---

## Waste decomposition

agenticmeter flags three categories of wasted spend:

**Retries.** Identical prompts sent within `retry_window_seconds` (default 60). Hash-based detection means semantically equivalent but textually different prompts are missed.

**Truncations.** Calls where the model hit the output token cap (`finish_reason == "length"` for OpenAI, `"max_tokens"` for Anthropic).

**Failures.** Calls that raised an exception. Cost is $0 (no tokens billed), but the failed attempt is still recorded for visibility.

```python
ledger.wasted_cost                     # 0.0697
ledger.wasted_cost_by_category()       # {'retries': 0.0557, 'truncated': 0.0140, 'failed': 0.0}
ledger.wasted_tokens()                 # token counts in the same shape
ledger.most_retried_prompts(n=5)       # [{'prompt_hash': 'abc123', 'calls': 4, 'cost': 0.042, 'models': ['gpt-4o']}, ...]
ledger.errors_by_type()                # {'RateLimitError': 2, 'TimeoutError': 1}
```

A single call can fall into multiple categories (a truncated retry, for example). To keep sums consistent (`sum(by_category) == wasted_cost`), priority is applied: **failed > retries > truncated**. Retries beat truncations because the retry is the duplicate-spend the user can fix; truncations are sometimes inherent to the workflow.

### Tuning retry detection

```python
Ledger(retry_window_seconds=300.0)  # 5-minute window; longer = more retries flagged
```

For batch jobs that legitimately reuse prompts hours apart, set this small or filter retries downstream from `ledger.calls`.

---

## Caching analysis

Both OpenAI and Anthropic support prompt caching. agenticmeter extracts cache fields from both providers automatically and shows:

- **Cache hit rate** — fraction of input tokens served from cache
- **Realized savings** — USD saved this run from cache hits
- **Missed opportunities** — repeated prompts not using caching, with dollar estimates of what caching would save

```python
ledger.cache_hit_rate()              # 0.0 to 1.0
ledger.realized_cache_savings()      # USD
ledger.caching_summary()             # full dict
ledger.missed_caching_opportunities()
# [{
#   'prompt_hash': 'abc123',
#   'provider': 'anthropic',
#   'model': 'claude-3-5-sonnet-20241022',
#   'count': 4,
#   'avg_input_tokens': 2500,
#   'current_cost': 0.0300,
#   'estimated_cached_cost': 0.0116,
#   'estimated_savings': 0.0184,
# }, ...]
```

### How extraction works

**OpenAI.** Returns `usage.prompt_tokens_details.cached_tokens`. We subtract this from `prompt_tokens` to get the uncached portion, then bill at the right rate. OpenAI caches automatically when prompts exceed ~1024 tokens.

**Anthropic.** Returns `usage.cache_read_input_tokens` and `usage.cache_creation_input_tokens` directly. These are separate from `input_tokens` (which is already uncached). Caching is opt-in via `cache_control` in the API request.

### How missed-opportunity detection works

After a run, the ledger groups successful, uncached calls by `prompt_hash`. For groups with at least 2 calls and average input ≥500 tokens, it estimates what caching would have cost (1 cache write + N-1 cache reads) and reports the delta vs current spend. Only opportunities with positive savings are returned (caching doesn't help every workload — short or one-shot prompts won't appear).

---

## Counterfactual costs

What would this exact workload have cost on a different model?

```python
ledger.counterfactuals()             # dict: 'provider/model' -> cost
ledger.cheapest_alternative()        # (provider, model, cost) tuple
ledger.savings_opportunities()       # list, sorted by biggest savings
```

`savings_opportunities()` is what the summary uses for the "Cheaper alternatives" section. It:

- Excludes the model you're currently using (zero savings)
- Excludes models that cost more (no opportunity)
- Dedupes dated variants with identical prices (e.g., `gpt-4o-mini` and `gpt-4o-mini-2024-07-18`)
- Returns top alternatives sorted by biggest savings

The math is straightforward — same tokens, different price — so it's accurate for OpenAI/Anthropic models. The catch: a cheaper model may produce different output quality. agenticmeter does not assess quality; it only does the cost arithmetic.

---

## Agent-loop detection

When a tagged agent makes 5+ LLM calls in one ledger, it's flagged as a potential loop:

```python
ledger.detect_agent_loops(min_calls=5)
# [{
#   'tag': 'invoice_agent',
#   'calls': 16,
#   'cost': 0.0480,
#   'input_tokens': 78400,
#   'avg_input_per_call': 4900,
#   'first_call_input_tokens': 580,
#   'last_call_input_tokens': 10640,
#   'context_growth_ratio': 18.3,
# }]
```

**Severity** is indicated by `context_growth_ratio`. In the summary:

- `stuck` — growth ≥2x; conversation history is accumulating, classic tool-loop signature
- `active` — growth <2x; many calls but flat context (could be legitimate parallel work)

The default threshold of 5 calls per tag is conservative. Tune via `min_calls=` for agents that legitimately do deep work:

```python
ledger.detect_agent_loops(min_calls=10)
```

---

## The LangChain callback

LangChain's `ChatOpenAI` and `ChatAnthropic` internally call the raw SDK, so our monkey-patches *fire* — but they fire on a streaming-internal path where token usage isn't exposed cleanly. The result without the callback: calls are counted but tokens come back as zero, so cost is $0.

The fix: a proper LangChain callback handler that extracts tokens from `LLMResult.llm_output['token_usage']` (where LangChain actually puts them) and suppresses the raw-SDK tracker so the call isn't double-counted.

```python
from agenticmeter import Ledger
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

with Ledger(budget="$1.00") as ledger:
    cb = ledger.as_langchain_callback()
    response = llm.invoke("hello", config={"callbacks": [cb]})

print(ledger.summary())
```

### Multi-agent attribution

```python
with Ledger(budget="$5.00", name="multi_agent") as ledger:
    cb = ledger.as_langchain_callback()

    with ledger.tag("researcher"):
        research = researcher_llm.invoke(q1, config={"callbacks": [cb]})
    with ledger.tag("writer"):
        draft = writer_llm.invoke(q2, config={"callbacks": [cb]})
    with ledger.tag("critic"):
        review = critic_llm.invoke(q3, config={"callbacks": [cb]})

print(ledger.by_tag())
```

### LangGraph

Pass the callback through `config`; it propagates down through all graph nodes and sub-agents automatically:

```python
graph = my_workflow.compile()

with Ledger(budget="$5.00") as ledger:
    cb = ledger.as_langchain_callback()
    config = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": [cb],  # propagates to every nested LLM call in the graph
    }
    result = graph.invoke({"input": "..."}, config=config)

print(ledger.summary())
```

For per-node attribution in LangGraph, wrap node implementations:

```python
def music_agent(state):
    with _tag("music_agent"):
        # all LLM calls inside get this tag
        return run_tool_agent(...)
```

See [`examples/langchain_multi_agent.py`](examples/langchain_multi_agent.py) for a runnable demo.

### What about other LangChain providers?

The callback uses LangChain's own response wrapper format, which is consistent across providers. Calls through `ChatBedrock`, `AzureChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatMistralAI`, etc. all flow through the same callback path and get tracked — provided the model name is in `prices.json` or our provider inference (which guesses from model name prefix) works.

If a Bedrock model isn't priced in the JSON, the call is still recorded (with $0 cost). Add the price to `prices.json` to get accurate cost numbers.

---

## Manual recording

For providers without auto-tracking, or for any non-LLM API call you want to associate with this ledger:

```python
ledger.record(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    input_tokens=1500,
    output_tokens=400,
    cached_read_tokens=0,       # tokens served from cache
    cached_write_tokens=0,      # tokens written to cache (Anthropic only)
    duration=2.3,               # seconds, optional
    finish_reason="stop",       # 'stop', 'length', 'max_tokens', 'error'
    failed=False,
    error=None,
    prompt_hash=None,           # for retry detection; pass a stable hash of messages
    metadata={"agent": "X"},    # arbitrary dict
    tags=None,                  # list of additional tags beyond the current tag stack
)
```

Common pattern: instrument your wrapper function once and every call goes through it:

```python
def my_llm_wrapper(model, messages, **kwargs):
    response = bedrock_client.invoke_model(...)
    ledger = get_current_ledger()
    if ledger is not None:
        ledger.record(
            provider="anthropic",
            model=model,
            input_tokens=response["usage"]["input_tokens"],
            output_tokens=response["usage"]["output_tokens"],
        )
    return response
```

---

## Pricing data

Prices live in [`src/agenticmeter/prices.json`](src/agenticmeter/prices.json). Per million tokens, USD, standard non-batch rates.

```json
{
  "openai": {
    "gpt-4o-mini": { "input": 0.15, "output": 0.60, "cached_read": 0.075 }
  },
  "anthropic": {
    "claude-3-5-haiku-20241022": {
      "input": 0.80, "output": 4.00,
      "cached_read": 0.08, "cached_write": 1.00
    }
  }
}
```

**Cache rates.** OpenAI cached reads are 50% of input. Anthropic cached reads are 10% of input; cache writes are 125% of input. Models without `cached_read`/`cached_write` keys fall back to the normal input rate (so workloads without caching are unaffected).

**Adding a model.** Edit the JSON. No code changes needed. Call `agenticmeter.reload_prices()` to pick up changes at runtime.

**Prefix matching.** Dated model strings (e.g., `gpt-4o-2024-12-15`) match the longest known prefix in prices.json. So adding the base name (`gpt-4o`) covers all date-suffixed variants.

**Verifying current prices.** Run a small real call and compare ledger cost to your provider dashboard. If they don't match within rounding, the JSON is stale — update it.

---

## Programmatic API reference

### `Ledger`

| Property / method | Returns | Description |
|---|---|---|
| `total_cost` | float | Sum of all successful call costs (USD) |
| `total_calls` | int | All calls including failed |
| `successful_calls` | int | Calls without `failed=True` |
| `failed_calls` | int | Calls with `failed=True` |
| `total_input_tokens` | int | Sum across calls |
| `total_output_tokens` | int | Sum across calls |
| `total_cached_read_tokens` | int | Sum across calls |
| `total_cached_write_tokens` | int | Sum across calls |
| `remaining_budget` | float or None | `budget - total_cost`, or None if no budget |
| `by_tag()` | dict | `{tag: cost}` using innermost tag |
| `calls_by_tag()` | dict | `{tag: count}` using innermost tag |
| `wasted_cost` | float | Sum of cost across waste categories |
| `wasted_cost_by_category()` | dict | `{'retries': X, 'truncated': Y, 'failed': Z}` |
| `wasted_tokens()` | dict | Same shape, token counts |
| `detect_retries()` | list[Call] | Calls flagged as retries |
| `detect_truncations()` | list[Call] | Calls that hit max_tokens |
| `detect_failed()` | list[Call] | Calls with `failed=True` |
| `most_retried_prompts(n=5)` | list[dict] | Top repeated prompts by total cost |
| `errors_by_type()` | dict | `{ErrorClass: count}` |
| `cache_hit_rate()` | float | 0.0 to 1.0 |
| `realized_cache_savings()` | float | USD saved by cache hits |
| `caching_summary()` | dict | Combined cache stats |
| `missed_caching_opportunities()` | list[dict] | Repeated uncached prompts |
| `counterfactuals()` | dict | `{'provider/model': cost}` for all known models |
| `cheapest_alternative()` | tuple or None | `(provider, model, cost)` |
| `savings_opportunities()` | list[tuple] | `(model_key, cost, savings_pct)` sorted |
| `detect_agent_loops(min_calls=5)` | list[dict] | Tagged agents with many calls |
| `summary()` | str | Human-readable report |
| `to_dict()` | dict | Serializable state |
| `to_json(indent=2)` | str | JSON string |
| `record(...)` | Call | Manually record a call |
| `tag(name)` | context manager | Push a tag onto the stack |

### `Call`

A `dataclass` representing one recorded LLM call:

```python
@dataclass
class Call:
    provider: str
    model: str
    input_tokens: int            # uncached input
    output_tokens: int
    cost: float                  # USD
    timestamp: float             # Unix seconds
    duration: float
    finish_reason: Optional[str]
    failed: bool
    error: Optional[str]
    prompt_hash: Optional[str]
    metadata: Dict[str, Any]
    tags: List[str]              # innermost-last
    cached_read_tokens: int
    cached_write_tokens: int
```

### Module-level helpers

| Function | Description |
|---|---|
| `get_current_ledger()` | Returns the currently active Ledger, or None |
| `calculate_cost(provider, model, input, output, cached_read=0, cached_write=0)` | Compute cost without a Ledger |
| `get_price(provider, model)` | Dict with `input`/`output`/`cached_read`/`cached_write` rates per 1M tokens |
| `list_models(provider=None)` | List of `(provider, model)` tuples |
| `reload_prices()` | Force reload of `prices.json` |
| `track_budget(budget, name=None, on_complete=None)` | Decorator form |

### Exceptions

| Class | When raised |
|---|---|
| `agenticmeterError` | Base class for all package exceptions |
| `BudgetExceeded` | When a tracked operation crosses the configured budget |

---

## Working with threads and async

**Async.** `asyncio` works out of the box. The Ledger uses `contextvars`, which propagate correctly to awaited code and to tasks created with `asyncio.create_task` (in Python 3.7+).

```python
async def run():
    with Ledger(budget="$1.00") as ledger:
        response = await async_client.chat.completions.create(...)
        results = await asyncio.gather(
            async_call_a(),  # the ledger is visible here
            async_call_b(),  # and here
        )
    print(ledger.summary())
```

**Threads.** `ThreadPoolExecutor` does **not** automatically propagate contextvars to worker threads. If you use threads, you must capture and re-run the context:

```python
import contextvars
from concurrent.futures import ThreadPoolExecutor

with Ledger() as ledger:
    with ledger.tag("parallel_work"):
        ctx = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=8) as executor:
            # WRONG: workers won't see the ledger or the tag
            # futures = [executor.submit(worker_fn, item) for item in items]

            # RIGHT: ctx.run wraps the call in the captured context
            futures = [
                executor.submit(ctx.run, worker_fn, item)
                for item in items
            ]
            results = [f.result() for f in futures]
```

Without `ctx.run`, calls made inside worker threads will see `get_current_ledger() == None` and won't be tracked. This is a standard Python pattern — the same applies to any contextvars-based library.

Automated propagation is planned for v0.4.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'agenticmeter'`** — your environment isn't the one where you installed it. If running in Jupyter, check `sys.executable` — it should point to the `.venv` you installed the package into. See [README "Notebook setup"](README.md) if needed.

**Ledger records 0 calls inside a `with` block.** Either the SDK isn't installed (e.g., you have `agenticmeter` but not `agenticmeter[openai]`), or the call is going through a path we don't patch (streaming, a custom HTTP client). Run `import agenticmeter; print(agenticmeter.__version__)` to confirm the package is current.

**LangChain calls show 0 tokens but the call count is right.** You haven't passed the callback through. Either add `config={"callbacks": [ledger.as_langchain_callback()]}` to your `.invoke()` call, or the call is happening on a streaming path. The LangChain callback handles non-streaming `.invoke()` and `.ainvoke()`.

**Calls double-counted: 2x the expected count.** You're running both the raw-SDK monkey-patch *and* the LangChain callback on the same call. Make sure your `with Ledger()` block contains only one or the other. If you're using LangChain, always pass the callback; the SDK patch is suppressed automatically when the callback is active.

**Tags don't appear in `by_tag()`.** Most common cause: the call happened in a thread spawned by `ThreadPoolExecutor` without `ctx.run`. See [Working with threads and async](#working-with-threads-and-async). Second most common: the call happened outside the `with ledger.tag(...)` block (e.g., the LangChain callback fired after the block exited; rare but possible with deeply nested async).

**`BudgetExceeded` doesn't fire even though I'm over budget.** The exception fires on the *next* `record()` after the cap is crossed. If no more calls happen, you'll see the overshoot in `ledger.total_cost` but no exception was raised. To assert no overshoot at the end, check `assert ledger.total_cost <= ledger.budget` after the block.

**Costs in the report don't match my provider dashboard.** Check `prices.json` against the live provider pricing page. Prices change quarterly-ish and the seed values may be stale. Fix the JSON; no reinstall needed.

**Found a bug or want a feature.** Open an issue with the smallest repro you can. The package is small enough that most bugs are debuggable from a 20-line script.