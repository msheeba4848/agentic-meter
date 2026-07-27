# Changelog

All notable changes to agenticmeter.

---

## [0.4.0] — Bedrock, custom wrappers, and threading

Adds native AWS Bedrock support and the `@track_llm_call` decorator for custom
wrappers. Threading propagation is now a one-line helper.

### Added

**AWS Bedrock native tracker**
- Monkey-patches `boto3.client('bedrock-runtime').invoke_model` and `.converse`
- Auto-detects response shape: Anthropic Claude, Meta Llama, Amazon Titan, Amazon Nova
- Extracts prompt-caching fields for Claude on Bedrock (`cache_read_input_tokens`, `cache_creation_input_tokens`)
- Preserves response body for downstream reading (uses a replay wrapper)
- Normalizes Bedrock model IDs (strips regional and vendor prefixes) so `prices.json` can match on the model name
- Records unknown model families with 0 cost rather than crashing
- Install: `pip install "agenticmeter[bedrock]"`

**Custom-wrapper `@track_llm_call` decorator**
- Instrument any function that calls an LLM provider without changing every call site
- Takes an `extract_usage(response, args, kwargs)` callback that returns provider/model/tokens
- Handles pass-through when no ledger is active
- Records failed calls with the exception message
- The one-line integration for custom Bedrock/Azure/Vertex/in-house wrappers

**Threading: `tracked_executor(executor)`**
- Wraps `ThreadPoolExecutor` (or any executor with `.submit`) so `contextvars` propagate to worker threads
- Captures the current context at wrap time — pushed tags become visible to workers
- Removes the need for manual `contextvars.copy_context()` + `ctx.run(...)` boilerplate

**Bedrock model prices**
- Added to `prices.json`: `claude-3-5-sonnet-20241022-v2:0`, `claude-3-5-haiku-20241022-v1:0`, `claude-3-opus-20240229-v1:0`, `claude-3-haiku-20240307-v1:0`, `claude-3-sonnet-20240229-v1:0`, `claude-3-5-sonnet-20240620-v1:0`
- Also added Anthropic Claude 4 (Opus/Sonnet) direct-API prices: `claude-opus-4-20250514`, `claude-sonnet-4-20250514`
- Meta Llama family: `llama3-70b-instruct-v1:0`, `llama3-8b-instruct-v1:0`, `llama3-1-*`, `llama3-2-*`
- Amazon Titan family: `titan-text-express-v1`, `titan-text-lite-v1`, `titan-text-premier-v1:0`
- Amazon Nova family: `nova-pro-v1:0`, `nova-lite-v1:0`, `nova-micro-v1:0`
- Added OpenAI GPT-4.1 series: `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`

### Fixed

- **Exception class rename**: `agenticmeterError` → `AgenticMeterError` (PascalCase, Python convention). The lowercase name remains as a deprecation alias through v0.x and will be removed in v1.0. Tracebacks and IDE autocomplete now show the conventional PascalCase form.
- **Stale docstring** in top-level `__init__.py` (still said "agentledger" from the pre-rename package name).
- **Cleaned up `trackers/__init__.py`**: previously a bad copy of the top-level `__init__.py` from the package rename. Now just a package marker.

### Provider support (v0.4)

| Provider | Auto-tracked | LangChain | Custom wrapper | Manual record |
|---|:---:|:---:|:---:|:---:|
| OpenAI | ✅ | ✅ | via `@track_llm_call` | ✅ |
| Anthropic | ✅ | ✅ | via `@track_llm_call` | ✅ |
| AWS Bedrock | ✅ (new) | ✅ | via `@track_llm_call` | ✅ |
| Azure OpenAI | via OpenAI tracker | ✅ | via `@track_llm_call` | ✅ |
| Google Vertex / Gemini | — | ✅ | via `@track_llm_call` | ✅ |
| Anything else | — | ✅ | via `@track_llm_call` | ✅ |

### Known limitations (still)

- Streaming responses (`stream=True`) still not auto-tracked. Planned for v0.5.
- Tool-use costs (web search, code sandboxes, computer use) not tracked separately from base LLM calls.
- Retry detection is text-hash based; semantically equivalent prompts with different wording are missed.

---

## [0.3.0] — Initial public release

The first version published to PyPI. Built and dogfooded on a LangChain
multi-agent pipeline and a production medical-policy compliance pipeline
before release.

### Added

**Core**
- `Ledger` class with in-process budget enforcement
- `track_budget` decorator
- `BudgetExceeded` exception
- Zero-dependency core

**Auto-tracking**
- OpenAI SDK monkey-patch (`Completions.create` sync + async)
- Anthropic SDK monkey-patch (`Messages.create` sync + async)
- LangChain callback handler with double-count suppression

**Reporting**
- Per-agent attribution via `Ledger.tag()`
- Counterfactual costs on all known models
- Waste decomposition (retries, truncations, failures) with dollar breakdowns
- Realized caching savings + missed caching opportunities
- Agent-loop detection with context-growth-ratio severity
- Human-readable `summary()`, structured `to_dict()`/`to_json()`

**Provider support**
- OpenAI, Anthropic, LangChain (any backend), Azure OpenAI via OpenAI tracker

### Known limitations (at 0.3)

- No native Bedrock/Vertex trackers (workaround: manual `ledger.record()`)
- Streaming not auto-tracked
- ThreadPoolExecutor requires manual `contextvars.copy_context()`
- The two v0.3 cosmetic bugs listed under 0.4's "Fixed" section
