# Changelog

All notable changes to agentledger. Format inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/) once the package is past v1.0.0. Pre-1.0, minor versions may include breaking changes.

---

## [0.3.0] — Initial public release

The first version published to PyPI. Built and dogfooded across multiple multi-agent pipelines (single-agent OpenAI scripts, LangChain ReAct agents, LangGraph multi-node workflows, and a production medical-policy compliance pipeline using Anthropic + Azure GPT-4.1 via Bedrock).

### Added

**Core**
- `Ledger` class: in-process budget enforcement, per-call tracking, summary reporting
- `track_budget` decorator: function-level wrapper around `Ledger`
- `BudgetExceeded` exception: raised the moment a tracked operation crosses the configured cap
- `Call` dataclass: structured representation of one LLM call with provider, model, tokens, cost, finish reason, error, prompt hash, tags, and cache fields
- Zero-dependency core; SDKs are optional extras

**Auto-tracking**
- OpenAI SDK monkey-patch: covers `Completions.create` and `AsyncCompletions.create` (non-streaming)
- Anthropic SDK monkey-patch: covers `Messages.create` and `AsyncMessages.create` (non-streaming)
- LangChain callback handler (`Ledger.as_langchain_callback()`): proper extraction from `LLMResult`, handles OpenAI and Anthropic shapes including newer `usage_metadata` paths
- Suppression mechanism: when LangChain callback is active, raw-SDK monkey-patch skips recording to prevent double-counting

**Budget enforcement**
- USD-string parsing (`"$2.00"`, `"2.00"`, `2.0` all accepted)
- Optional `on_budget_exceeded` callback for alerts before the exception fires
- `remaining_budget` property for in-flight progress bars

**Per-agent attribution**
- `Ledger.tag(name)` context manager for tagging calls
- Nested tags compose; innermost is used for primary attribution
- `by_tag()`, `calls_by_tag()` reporting
- Tags stamped onto `Call` records for downstream slicing

**Counterfactual costs**
- `counterfactuals()`: cost of this workload on every known model
- `cheapest_alternative()`: single best alternative
- `savings_opportunities()`: sorted list of alternatives that would save money, with dedupe of dated variants and exclusion of equally-or-more-expensive options
- "Cheaper alternatives" section in summary, with "save X%" framing

**Waste decomposition**
- Detection of retries (same prompt hash within window), truncations (finish_reason `length` or `max_tokens`), and failed calls
- `wasted_cost_by_category()`: priority-deduped dollar breakdown (failed > retries > truncated)
- `wasted_tokens()`: same shape, token counts
- `most_retried_prompts(n=5)`: worst offenders by total cost
- `errors_by_type()`: failure categorization by error class
- Detailed waste section in summary

**Caching analysis**
- Cache field extraction: OpenAI (`prompt_tokens_details.cached_tokens`) and Anthropic (`cache_read_input_tokens`, `cache_creation_input_tokens`)
- Cache pricing in `prices.json`: OpenAI 50% of input for cached reads; Anthropic 10% read / 125% write
- `cache_hit_rate()`, `realized_cache_savings()`, `caching_summary()` reporting
- `missed_caching_opportunities()`: flags repeated uncached prompts with dollar estimates of savings if caching were enabled
- Caching + Missed-opportunities sections in summary

**Agent-loop detection**
- `detect_agent_loops(min_calls=5)`: surfaces tagged agents with suspiciously many calls
- Context-growth ratio (first-call vs last-call input tokens) as severity signal
- "Agent loops detected" section in summary with `stuck`/`active` labels based on growth ≥2x

**Pricing data**
- Standalone JSON file at `src/agentledger/prices.json` — no code changes needed to update prices
- Coverage: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo, o1, o1-mini, o3-mini, claude-3-opus, claude-3-5-sonnet, claude-3-5-haiku, claude-3-haiku, claude-3-sonnet, plus dated variants
- `reload_prices()` for runtime updates
- Prefix matching for dated model strings

**Reporting**
- `summary()`: human-readable multi-section report
- `to_dict()` / `to_json()`: structured serialization
- Provider breakdown, per-tag breakdown, waste decomposition, caching analysis, missed opportunities, cheaper alternatives — all in one summary

**Examples**
- `examples/basic_usage.py`: minimal manual recording demo
- `examples/runaway_agent.py`: demonstrates budget enforcement stopping a bad loop
- `examples/langchain_multi_agent.py`: real LangChain multi-agent demo with per-agent attribution

### Provider support

| Provider | v0.3 |
|---|---|
| OpenAI | auto-tracked |
| Anthropic | auto-tracked |
| LangChain (any backend) | callback handler |
| Azure OpenAI | works via OpenAI tracker (non-streaming verified) |
| Bedrock | manual `record()` only |
| Vertex AI / Gemini | manual `record()` only |

### Known limitations

- Streaming responses (`stream=True`) are not auto-tracked
- Bedrock, Vertex, and Gemini lack native auto-trackers
- `ThreadPoolExecutor` requires manual `contextvars.copy_context()` for tag propagation
- Retry detection is text-hash based; semantically equivalent prompts with different wording are not flagged
- Tool-call costs (web search, code execution sandboxes) are not auto-tracked

---

## [Unreleased / Planned for 0.4]

- Streaming response auto-tracking (OpenAI + Anthropic)
- Native AWS Bedrock tracker (`boto3` `invoke_model` and `converse`)
- Native Azure OpenAI tracker (full coverage, currently partial via OpenAI tracker)
- Native Google Gemini tracker
- `@track_llm_call` decorator for custom wrappers (covers any provider via user-supplied extractor)
- Automatic ContextVar propagation through `ThreadPoolExecutor` and `asyncio.create_task`
- CLI: `agentledger analyze <ledger.json>` for retrospective reports
- Pre-flight cost estimation from saved run history
- Cost regression testing fixture for pytest