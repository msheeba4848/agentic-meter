# agenticmeter (lol)

**FinOps for LLM agents.** Budget caps that actually stop runaway calls. Waste decomposition that shows where the money went. Caching analysis that tells you what to fix. Agent-loop detection for multi-agent pipelines.

Drop-in for OpenAI, Anthropic, and AWS Bedrock. Works with LangChain. One-line integration for any custom wrapper. Zero runtime dependencies in core. No backend, no account, no telemetry.

```python
from agenticmeter import Ledger
import openai

client = openai.OpenAI()

with Ledger(budget="$2.00") as ledger:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
    )

print(ledger.summary())
```

---

## Why this exists

Every other tool in this space — tokencost, LangSmith, Langfuse, Helicone, LiteLLM's cost tracking — tells you what you *spent*. None of them tell you:

- That 86% of your spend was waste (retries on the same prompt, truncated outputs, failed calls)
- That the same workload on gpt-4o-mini would have cost 94% less
- That your invoice_agent is stuck in a tool-calling loop, eating $0.30 per run before giving up
- That a 2,500-token system prompt is being sent uncached on every iteration when caching would cut that input cost by 90%

agenticmeter does. It's the layer that sits between your code and your monthly OpenAI/Anthropic bill, and it turns vague "this feels expensive" feelings into specific dollar-denominated decisions.

It also enforces budgets *in-process*. Other tools warn you tomorrow. agenticmeter raises `BudgetExceeded` on the next call, the moment you cross the line — so a runaway agent stops at $2.00 instead of $200.

## Install

```bash
pip install agenticmeter              # core only, zero deps
pip install agenticmeter[openai]      # + OpenAI auto-tracking
pip install agenticmeter[anthropic]   # + Anthropic auto-tracking
pip install agenticmeter[bedrock]     # + AWS Bedrock auto-tracking (new in v0.4)
pip install agenticmeter[langchain]   # + LangChain callback handler
pip install agenticmeter[all]         # everything
```

Core has zero runtime dependencies. SDKs are optional extras — install only what you use.

## Quickstart

### OpenAI

```python
from agenticmeter import Ledger
from openai import OpenAI

client = OpenAI()

with Ledger(budget="$1.00", name="my_agent") as ledger:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )

print(ledger.summary())
```

### Anthropic

```python
from agenticmeter import Ledger
from anthropic import Anthropic

client = Anthropic()

with Ledger(budget="$1.00") as ledger:
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hello!"}],
    )

print(ledger.summary())
```

### LangChain (any provider it supports)

LangChain wraps responses differently than the raw SDKs, so it has its own callback handler that extracts tokens correctly and handles per-agent attribution.

```python
from agenticmeter import Ledger
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

with Ledger(budget="$1.00") as ledger:
    cb = ledger.as_langchain_callback()

    with ledger.tag("researcher"):
        result_a = llm.invoke("research topic A", config={"callbacks": [cb]})
    with ledger.tag("writer"):
        result_b = llm.invoke("write about it", config={"callbacks": [cb]})

print(ledger.summary())
```

### AWS Bedrock (new in v0.4)

Any `boto3.client('bedrock-runtime')` call inside a `Ledger` is auto-tracked. Works for Claude, Llama, Titan, and Nova.

```python
import boto3
import json
from agenticmeter import Ledger

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

with Ledger(budget="$1.00", name="bedrock_test") as ledger:
    response = bedrock.invoke_model(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Hello!"}],
        }),
    )

print(ledger.summary())
```

Prompt caching fields for Claude on Bedrock (`cache_read_input_tokens`, `cache_creation_input_tokens`) are extracted automatically. Response bodies remain readable after tracking — the tracker uses a replay wrapper so downstream code that does `response['body'].read()` still works.

### Custom wrappers (Azure, Vertex, in-house routers)

For providers or internal wrappers not natively supported, use the `@track_llm_call` decorator. Instrument once, track forever.

```python
from agenticmeter import track_llm_call, Ledger

def extract_usage(response, args, kwargs):
    return {
        "provider": "anthropic",
        "model": kwargs["model"],
        "input_tokens": response["usage"]["input_tokens"],
        "output_tokens": response["usage"]["output_tokens"],
    }

@track_llm_call(extract_usage=extract_usage)
def my_bedrock_wrapper(prompt, model, max_tokens):
    # Your existing wrapper that calls boto3 or any provider
    return call_bedrock_or_whatever(...)

with Ledger(budget="$1.00") as ledger:
    my_bedrock_wrapper("hi", model="claude-3-5-sonnet-20241022-v2:0", max_tokens=100)

print(ledger.summary())
```

## What a report looks like

Real output from a multi-agent customer support pipeline:

```
Ledger Summary (customer_support_agent)
=======================================
Total spent:      $0.0807 / $5.0000
Total calls:      9  (8 ok, 1 failed)
Total tokens:     13,650 in / 4,352 out

By provider:
  anthropic     $0.0420
  openai        $0.0387

By tag:
  rag_agent     $0.0480  (4 calls)
  writer        $0.0327  (2 calls)
  classifier    $0.0000  (3 calls)

Agent loops detected:
  rag_agent     16 calls, $0.0480  (context grew 22.0x: 500 -> 11,000 tokens, stuck)

Waste:            $0.0697 (86.4% of spend)
  retries       $0.0557  (11,648 tokens)
  truncated     $0.0140  (2,524 tokens)
  Top repeated prompts:
    rag_query    x4  $0.0420
    answer       x2  $0.0382
  Errors:       RateLimitError=1

Caching:          11.7% hit rate, saved $0.0001
  Read tokens:  1,800

Missed caching opportunities: ~$0.0203 potential savings
  rag_query    x4 @ ~2,500 tokens could save $0.0184
  answer       x2 @ ~1,500 tokens could save $0.0019

Cheaper alternatives (same workload):
  openai/gpt-4o-mini                         $0.0047  (save 94%)
  anthropic/claude-3-haiku-20240307          $0.0089  (save 89%)
  openai/gpt-3.5-turbo                       $0.0134  (save 83%)
```

Five things you can act on immediately, all from one report.

## Core features

**Hard budget enforcement.** `BudgetExceeded` is raised the moment a tracked operation crosses your cap. Runaway agents fail fast instead of burning credit.

**Per-agent attribution via tags.** Wrap blocks with `ledger.tag("name")` and the report shows where the money went per agent. Essential for multi-agent pipelines.

**Waste decomposition.** Retries, truncations, and failures are tracked separately with dollar amounts and token counts. Top repeated prompts and error types are surfaced automatically.

**Counterfactual costs / cheaper alternatives.** Every report shows what the same workload would have cost on smaller or different models, sorted by biggest savings first.

**Prompt-caching analysis.** Realized savings from cache hits are extracted from OpenAI and Anthropic responses. Missed opportunities — repeated long prompts that aren't using caching — are flagged with dollar estimates of what caching would save.

**Agent-loop detection.** When a tagged agent makes many LLM calls in one ledger run, the package flags it as a likely tool-calling loop, with context-growth ratio as severity.

**Three ways to integrate** — context manager (recommended), decorator (`@track_budget`), or manual `ledger.record()` for providers without auto-tracking.

## Three ways to use it

### Context manager (recommended)

```python
with Ledger(budget="$5.00", name="nightly_batch") as ledger:
    run_my_agent()

print(ledger.summary())
```

### Decorator

```python
from agenticmeter import track_budget

@track_budget("$0.50", on_complete=lambda l: print(l.summary()))
def answer_question(q):
    ...
```

### Manual recording

For providers we don't auto-track yet (Bedrock, Azure, Vertex AI, custom wrappers):

```python
with Ledger(budget="$1.00") as ledger:
    response = my_bedrock_call(...)
    ledger.record(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        input_tokens=response["usage"]["input_tokens"],
        output_tokens=response["usage"]["output_tokens"],
    )
```

## Provider support

| Provider | Auto-tracked | LangChain | Custom wrapper | Manual record |
|---|:---:|:---:|:---:|:---:|
| OpenAI | ✅ | ✅ | via `@track_llm_call` | ✅ |
| Anthropic | ✅ | ✅ | via `@track_llm_call` | ✅ |
| AWS Bedrock | ✅ (v0.4) | ✅ | via `@track_llm_call` | ✅ |
| Azure OpenAI | via OpenAI tracker | ✅ | via `@track_llm_call` | ✅ |
| Google Vertex / Gemini | — | ✅ | via `@track_llm_call` | ✅ |
| Cohere, Mistral, everything else | — | ✅ | via `@track_llm_call` | ✅ |

Bedrock covers Anthropic Claude, Meta Llama, Amazon Titan, and Amazon Nova via `invoke_model` and `converse`. Use `pip install "agenticmeter[bedrock]"` to install boto3 as well.

Native Vertex / Gemini trackers planned for v0.5. Until then, the `@track_llm_call` decorator handles them in one line.

## How it compares

|  | agenticmeter | tokencost | LangSmith | Langfuse | LiteLLM |
|---|:---:|:---:|:---:|:---:|:---:|
| Cost calculation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Budget *enforcement* in-process | ✅ | — | — | — | proxy only |
| Counterfactual costs (savings on other models) | ✅ | — | — | — | — |
| Waste decomposition (retries, truncations, failed) | ✅ | — | partial | partial | — |
| Missed caching detection | ✅ | — | — | — | — |
| Realized cache savings | ✅ | — | — | — | — |
| Agent-loop detection | ✅ | — | — | — | — |
| Per-agent tagging | ✅ | — | ✅ | ✅ | ✅ |
| Works offline / no account | ✅ | ✅ | — | self-host | ✅ |
| Zero core dependencies | ✅ | — | — | — | — |

## Limitations (v0.4)

- **Streaming responses aren't auto-tracked yet.** Non-streaming `create()` / `invoke_model()` calls only. Streaming planned for v0.5.
- **Native Vertex / Gemini trackers still pending.** Use `@track_llm_call` for now.
- **Prompt retry detection is text-hash based.** Semantically equivalent prompts with different wording are missed.
- **Tool-use costs (web search, code sandboxes) aren't tracked separately** from base LLM calls.

## Programmatic API

Everything in `summary()` is also available as data:

```python
ledger.total_cost                       # float, USD
ledger.remaining_budget                 # float or None
ledger.total_calls                      # int
ledger.successful_calls                 # int
ledger.failed_calls                     # int
ledger.total_input_tokens               # int
ledger.total_output_tokens              # int

ledger.by_tag()                         # dict: tag -> cost
ledger.calls_by_tag()                   # dict: tag -> count
ledger.wasted_cost                      # float
ledger.wasted_cost_by_category()        # dict: retries/truncated/failed -> cost
ledger.wasted_tokens()                  # dict: same shape, tokens
ledger.most_retried_prompts(n=5)        # list of {prompt_hash, calls, cost}
ledger.errors_by_type()                 # dict: ErrorClass -> count

ledger.cache_hit_rate()                 # 0.0 to 1.0
ledger.realized_cache_savings()         # float, USD
ledger.caching_summary()                # dict
ledger.missed_caching_opportunities()   # list of actionable opportunities

ledger.detect_agent_loops()             # list of {tag, calls, growth_ratio, ...}
ledger.savings_opportunities()          # list of cheaper alternatives
ledger.counterfactuals()                # dict: model -> cost for this workload
ledger.cheapest_alternative()           # (provider, model, cost) tuple

ledger.to_dict()                        # full structured data
ledger.to_json()                        # JSON string
```

## Threading and async

Async works out of the box — `contextvars` propagate through awaited code and `asyncio.create_task`.

Threads need one line. `ThreadPoolExecutor` doesn't propagate `contextvars` to worker threads by default, so LLM calls inside workers would see `get_current_ledger() == None` and go untracked. Wrap the executor:

```python
from concurrent.futures import ThreadPoolExecutor
from agenticmeter import Ledger, tracked_executor

with Ledger() as ledger:
    with ledger.tag("parallel_work"):
        with tracked_executor(ThreadPoolExecutor(max_workers=8)) as ex:
            futures = [ex.submit(worker_fn, item) for item in items]
            results = [f.result() for f in futures]
```

`tracked_executor` captures the current context at wrap time, so any tags active when you call it propagate to workers. Wrap inside the tag block you want workers to inherit.

## Pricing data

Prices live in [`src/agenticmeter/prices.json`](src/agenticmeter/prices.json). Standard non-batch rates. Cache rates included: OpenAI cached reads at 50% of input, Anthropic at 10% read / 125% write.

To add a model or update a price, edit the JSON — no code changes.

## License

MIT.

## Status

Alpha (v0.4.0). APIs may shift. Battle-tested on real multi-agent pipelines including OpenAI, Anthropic, and Bedrock. File issues with rough edges — they're useful.

## Roadmap

See [`changelog.md`](changelog.md) for what's shipped. Coming in v0.5:
- Streaming response auto-tracking
- Native Google Vertex / Gemini tracker
- Native Azure OpenAI tracker with full parity
- CLI for retrospective analysis of saved runs
- Cost regression testing fixture for pytest