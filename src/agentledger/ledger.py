"""The Ledger - core tracking, budget enforcement, and reporting."""
from __future__ import annotations

import contextvars
import functools
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union

from .pricing import calculate_cost, list_models
from .exceptions import BudgetExceeded


# ContextVar so nested contexts and async tasks each see the right ledger.
_current_ledger: contextvars.ContextVar[Optional["Ledger"]] = contextvars.ContextVar(
    "current_ledger", default=None
)


@dataclass
class Call:
    """A single recorded LLM call."""
    provider: str
    model: str
    input_tokens: int                     # Uncached input tokens (billed at full rate)
    output_tokens: int
    cost: float
    timestamp: float
    duration: float = 0.0
    finish_reason: Optional[str] = None
    failed: bool = False
    error: Optional[str] = None
    prompt_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    # Prompt-caching fields. cached_read_tokens were served from cache (cheap).
    # cached_write_tokens were written to cache (Anthropic only; ~25% surcharge).
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0


def _parse_budget(budget: Union[str, int, float, None]) -> Optional[float]:
    """Accept '$2.00', '2.00', 2.0, etc."""
    if budget is None:
        return None
    if isinstance(budget, (int, float)):
        return float(budget)
    if isinstance(budget, str):
        cleaned = budget.replace("$", "").replace(",", "").strip()
        return float(cleaned)
    raise ValueError(f"Cannot parse budget value: {budget!r}")


class Ledger:
    """Tracks LLM API calls, enforces budgets, reports waste & counterfactual costs.

    Primary usage as a context manager:

        with Ledger(budget="$2.00") as ledger:
            response = openai_client.chat.completions.create(...)
            response = anthropic_client.messages.create(...)
        print(ledger.summary())

    Inside the `with` block, OpenAI and Anthropic SDK calls are automatically
    intercepted and recorded. Outside the block, the SDKs behave normally.

    Args:
        budget: Optional spending cap in USD. Accepts '$2.00', '2.00', or 2.0.
            When exceeded, the *next* tracked call raises BudgetExceeded.
        name: Optional label for this ledger (used in reports).
        retry_window_seconds: Window for retry detection. Identical prompts
            within this window are flagged as retries.
        on_budget_exceeded: Optional callback fired before the exception is
            raised. Useful for alerting.
    """

    def __init__(
        self,
        budget: Union[str, int, float, None] = None,
        name: Optional[str] = None,
        retry_window_seconds: float = 60.0,
        on_budget_exceeded: Optional[Callable[["Ledger"], None]] = None,
    ):
        self.budget = _parse_budget(budget)
        self.name = name
        self.retry_window = retry_window_seconds
        self.on_budget_exceeded = on_budget_exceeded
        self.calls: List[Call] = []
        self._token: Optional[contextvars.Token] = None
        self._tag_stack: List[str] = []
        # SDK suppression: when > 0, raw-SDK monkey-patches skip recording.
        # Set by higher-level integrations (e.g. LangChain callback) to prevent
        # double-counting. Attached to the Ledger (not a ContextVar) because
        # LangChain runs callbacks in a copied context.
        self._suppress_sdk: int = 0
        self._suppress_lock = threading.Lock()

    # ---- Context manager ----

    def __enter__(self) -> "Ledger":
        self._token = _current_ledger.set(self)
        self._apply_patches()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._token is not None:
            _current_ledger.reset(self._token)
            self._token = None
        return False  # don't suppress exceptions

    def _apply_patches(self) -> None:
        """Install monkey-patches on OpenAI and Anthropic SDKs if available.

        Patches are applied lazily on first context entry and persist across
        contexts (the patched functions check the current ContextVar). This
        keeps overhead near zero when no ledger is active.
        """
        try:
            from .trackers.openai_tracker import patch_openai
            patch_openai()
        except ImportError:
            pass
        try:
            from .trackers.anthropic_tracker import patch_anthropic
            patch_anthropic()
        except ImportError:
            pass

    # ---- Recording ----

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration: float = 0.0,
        finish_reason: Optional[str] = None,
        failed: bool = False,
        error: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        cached_read_tokens: int = 0,
        cached_write_tokens: int = 0,
    ) -> Call:
        """Manually record a call. Use this for providers without auto-tracking.

        ``input_tokens`` should be the UNCACHED input tokens (billed at full rate).
        ``cached_read_tokens`` are input tokens served from cache.
        ``cached_write_tokens`` are input tokens written to cache (Anthropic-style).
        """
        if failed:
            cost = 0.0
        else:
            cost = calculate_cost(
                provider, model, input_tokens, output_tokens,
                cached_read_tokens=cached_read_tokens,
                cached_write_tokens=cached_write_tokens,
            )
        # Combine any explicit tags with the current tag stack
        effective_tags = list(self._tag_stack)
        if tags:
            effective_tags.extend(tags)
        call = Call(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=time.time(),
            duration=duration,
            finish_reason=finish_reason,
            failed=failed,
            error=error,
            prompt_hash=prompt_hash,
            metadata=metadata or {},
            tags=effective_tags,
            cached_read_tokens=cached_read_tokens,
            cached_write_tokens=cached_write_tokens,
        )
        self.calls.append(call)
        self._check_budget()
        return call

    def _check_budget(self) -> None:
        if self.budget is None:
            return
        if self.total_cost > self.budget:
            if self.on_budget_exceeded:
                try:
                    self.on_budget_exceeded(self)
                except Exception:
                    pass  # don't let callback failures hide the budget error
            raise BudgetExceeded(self.total_cost, self.budget)

    # ---- Aggregates ----

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls if not c.failed)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def successful_calls(self) -> int:
        return sum(1 for c in self.calls if not c.failed)

    @property
    def failed_calls(self) -> int:
        return sum(1 for c in self.calls if c.failed)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def remaining_budget(self) -> Optional[float]:
        if self.budget is None:
            return None
        return max(0.0, self.budget - self.total_cost)

    # ---- Waste detection ----

    def detect_retries(self) -> List[Call]:
        """Calls that look like retries: same prompt hash within retry_window."""
        retries: List[Call] = []
        seen: Dict[str, float] = {}
        for call in self.calls:
            if call.prompt_hash is None:
                continue
            last = seen.get(call.prompt_hash)
            if last is not None and (call.timestamp - last) <= self.retry_window:
                retries.append(call)
            seen[call.prompt_hash] = call.timestamp
        return retries

    def detect_truncations(self) -> List[Call]:
        """Calls that ran into the output token cap."""
        return [
            c for c in self.calls
            if c.finish_reason in ("length", "max_tokens")
        ]

    def detect_failed(self) -> List[Call]:
        return [c for c in self.calls if c.failed]

    @property
    def wasted_cost(self) -> float:
        """Total cost across retries + truncations + failed calls (deduped)."""
        wasted_ids = set()
        for c in self.detect_retries():
            wasted_ids.add(id(c))
        for c in self.detect_truncations():
            wasted_ids.add(id(c))
        for c in self.detect_failed():
            wasted_ids.add(id(c))
        return sum(c.cost for c in self.calls if id(c) in wasted_ids)

    # ---- Counterfactuals ----

    def counterfactual_cost(self, target_provider: str, target_model: str) -> float:
        """What this ledger's workload would have cost on a different model."""
        total = 0.0
        for call in self.calls:
            if call.failed:
                continue
            total += calculate_cost(
                target_provider, target_model, call.input_tokens, call.output_tokens
            )
        return total

    def counterfactuals(self) -> Dict[str, float]:
        """All known models' costs for this workload, keyed 'provider/model'."""
        return {
            f"{p}/{m}": self.counterfactual_cost(p, m)
            for p, m in list_models()
        }

    def cheapest_alternative(self) -> Optional[tuple]:
        """Return (provider, model, cost) of the cheapest alternative, or None."""
        cf = self.counterfactuals()
        nonzero = [(k, v) for k, v in cf.items() if v > 0]
        if not nonzero:
            return None
        cheapest_key, cheapest_cost = min(nonzero, key=lambda x: x[1])
        provider, model = cheapest_key.split("/", 1)
        return (provider, model, cheapest_cost)

    def savings_opportunities(self) -> List[tuple]:
        """Models that would save money on this workload.

        Returns a list of (model_key, alt_cost, savings_pct) tuples, sorted by
        biggest savings first. Excludes models that cost the same or more than
        the current workload, and dedupes dated variants that share a price
        (e.g., 'gpt-4o-mini' and 'gpt-4o-mini-2024-07-18').

        Empty list means you're already on a cost-optimal model.
        """
        if self.total_cost <= 0:
            return []
        cf = self.counterfactuals()
        alternatives: List[tuple] = []
        seen_costs = set()
        for model_key, cost in cf.items():
            if cost <= 0 or cost >= self.total_cost:
                continue
            # Dedupe identical prices - keeps the first (typically cleaner-named) entry
            rounded = round(cost, 8)
            if rounded in seen_costs:
                continue
            seen_costs.add(rounded)
            savings_pct = (1 - cost / self.total_cost) * 100
            alternatives.append((model_key, cost, savings_pct))
        # Biggest savings first
        alternatives.sort(key=lambda x: -x[2])
        return alternatives

    # ---- Tagging (per-agent / per-feature attribution) ----

    @contextmanager
    def tag(self, name: str):
        """Push a tag onto the stack for the duration of this block.

        Every call recorded inside gets stamped with this tag. Nested tags
        compose - a call inside `with ledger.tag('pipeline'): with ledger.tag('researcher'):`
        gets both tags but is attributed to the innermost ('researcher') by by_tag().

        Example:
            with Ledger() as l:
                with l.tag('researcher'):
                    researcher.invoke(...)
                with l.tag('writer'):
                    writer.invoke(...)
            print(l.by_tag())
        """
        self._tag_stack.append(name)
        try:
            yield self
        finally:
            self._tag_stack.pop()

    def by_tag(self) -> Dict[str, float]:
        """Cost breakdown by innermost (most specific) tag.

        Calls without any tag are grouped under '_untagged'. The sum across
        all keys equals total_cost (single-attribution, no double counting).
        """
        out: Dict[str, float] = {}
        for c in self.calls:
            if c.failed:
                continue
            tag = c.tags[-1] if c.tags else "_untagged"
            out[tag] = out.get(tag, 0.0) + c.cost
        return out

    def calls_by_tag(self) -> Dict[str, int]:
        """Call count by innermost tag."""
        out: Dict[str, int] = {}
        for c in self.calls:
            tag = c.tags[-1] if c.tags else "_untagged"
            out[tag] = out.get(tag, 0) + 1
        return out

    # ---- LangChain integration ----

    def as_langchain_callback(self):
        """Return a LangChain BaseCallbackHandler that records into this Ledger.

        Requires langchain-core to be installed:
            pip install 'agentledger[langchain]'

        Example:
            with Ledger(budget='$1.00') as ledger:
                cb = ledger.as_langchain_callback()
                response = llm.invoke('hello', config={'callbacks': [cb]})
        """
        from .langchain import AgentLedgerCallback
        return AgentLedgerCallback()

    # ---- Waste decomposition ----

    def wasted_cost_by_category(self) -> Dict[str, float]:
        """Dollar breakdown of waste across retries, truncations, failures.

        A single call can fall into multiple categories (e.g. a truncated retry).
        Cost is attributed to the *most specific waste signal* in this priority:
        failed > retries > truncated. Retries beat truncation because the retry
        is the duplicate spend — the bad behavior the user can fix. Sums always
        equal `wasted_cost`.
        """
        out = {"failed": 0.0, "retries": 0.0, "truncated": 0.0}
        retry_ids = {id(c) for c in self.detect_retries()}
        trunc_ids = {id(c) for c in self.detect_truncations()}
        for c in self.calls:
            if c.failed:
                out["failed"] += c.cost
            elif id(c) in retry_ids:
                out["retries"] += c.cost
            elif id(c) in trunc_ids:
                out["truncated"] += c.cost
        return out

    def wasted_tokens(self) -> Dict[str, int]:
        """Token breakdown (input+output) across waste categories."""
        out = {"failed": 0, "retries": 0, "truncated": 0}
        retry_ids = {id(c) for c in self.detect_retries()}
        trunc_ids = {id(c) for c in self.detect_truncations()}
        for c in self.calls:
            tokens = c.input_tokens + c.output_tokens
            if c.failed:
                out["failed"] += tokens
            elif id(c) in retry_ids:
                out["retries"] += tokens
            elif id(c) in trunc_ids:
                out["truncated"] += tokens
        return out

    def most_retried_prompts(self, n: int = 5) -> List[Dict]:
        """Top N prompts ranked by total cost spent on duplicates of them.

        Returns dicts: {prompt_hash, calls, cost, models}.
        """
        groups: Dict[str, List[Call]] = {}
        for c in self.calls:
            if c.prompt_hash is None:
                continue
            groups.setdefault(c.prompt_hash, []).append(c)
        offenders = [
            {
                "prompt_hash": h,
                "calls": len(calls),
                "cost": sum(c.cost for c in calls),
                "models": sorted({c.model for c in calls}),
            }
            for h, calls in groups.items()
            if len(calls) >= 2  # only repeats are interesting
        ]
        offenders.sort(key=lambda x: -x["cost"])
        return offenders[:n]

    def errors_by_type(self) -> Dict[str, int]:
        """Count failed calls grouped by error type (first word of error message)."""
        out: Dict[str, int] = {}
        for c in self.calls:
            if not c.failed:
                continue
            key = (c.error or "unknown").split(":")[0].strip() or "unknown"
            out[key] = out.get(key, 0) + 1
        return out

    # ---- Caching analysis ----

    @property
    def total_cached_read_tokens(self) -> int:
        return sum(c.cached_read_tokens for c in self.calls)

    @property
    def total_cached_write_tokens(self) -> int:
        return sum(c.cached_write_tokens for c in self.calls)

    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache. 0.0 to 1.0."""
        cached = self.total_cached_read_tokens
        total = cached + self.total_input_tokens
        return cached / total if total > 0 else 0.0

    def realized_cache_savings(self) -> float:
        """USD saved by cache HITS this run (vs. paying full input price)."""
        from .pricing import get_price
        savings = 0.0
        for c in self.calls:
            if c.cached_read_tokens <= 0:
                continue
            price = get_price(c.provider, c.model)
            if not price:
                continue
            full_rate = price["input"]
            cached_rate = price.get("cached_read", full_rate)
            savings += c.cached_read_tokens * (full_rate - cached_rate) / 1_000_000
        return savings

    def caching_summary(self) -> Dict[str, Any]:
        """Aggregate caching stats for this ledger."""
        return {
            "cache_hit_rate": self.cache_hit_rate(),
            "cached_read_tokens": self.total_cached_read_tokens,
            "cached_write_tokens": self.total_cached_write_tokens,
            "realized_savings": self.realized_cache_savings(),
        }

    def missed_caching_opportunities(
        self, min_repeat: int = 2, min_input_tokens: int = 500
    ) -> List[Dict]:
        """Prompts repeated without caching that could plausibly benefit.

        Heuristic: group successful, *uncached* calls by prompt_hash; for groups
        with at least `min_repeat` occurrences and average input >= `min_input_tokens`,
        estimate what caching would have saved (1 cache write + N-1 cache reads).

        Returns dicts sorted by estimated savings (largest first), with keys:
        prompt_hash, provider, model, count, avg_input_tokens, current_cost,
        estimated_cached_cost, estimated_savings.
        """
        from .pricing import get_price

        groups: Dict[str, List[Call]] = {}
        for c in self.calls:
            if c.failed or c.prompt_hash is None:
                continue
            if c.cached_read_tokens > 0 or c.cached_write_tokens > 0:
                continue  # already using caching
            groups.setdefault(c.prompt_hash, []).append(c)

        opportunities: List[Dict] = []
        for h, calls in groups.items():
            if len(calls) < min_repeat:
                continue
            avg_input = sum(c.input_tokens for c in calls) / len(calls)
            if avg_input < min_input_tokens:
                continue
            sample = calls[0]
            price = get_price(sample.provider, sample.model)
            if not price:
                continue
            current_cost = sum(c.cost for c in calls)
            # Estimate: 1st call = cache write (write rate), rest = cache reads.
            input_rate = price["input"]
            cw_rate = price.get("cached_write", input_rate)
            cr_rate = price.get("cached_read", input_rate)
            total_input = sum(c.input_tokens for c in calls)
            total_output = sum(c.output_tokens for c in calls)
            # First call's input billed at write rate; remaining N-1 calls' input
            # billed at read rate. Output is unchanged.
            first_input = calls[0].input_tokens
            rest_input = total_input - first_input
            estimated_cached_cost = (
                first_input * cw_rate
                + rest_input * cr_rate
                + total_output * price["output"]
            ) / 1_000_000
            savings = current_cost - estimated_cached_cost
            if savings <= 0:
                continue  # caching wouldn't actually help here
            opportunities.append({
                "prompt_hash": h,
                "provider": sample.provider,
                "model": sample.model,
                "count": len(calls),
                "avg_input_tokens": int(avg_input),
                "current_cost": current_cost,
                "estimated_cached_cost": estimated_cached_cost,
                "estimated_savings": savings,
            })
        opportunities.sort(key=lambda x: -x["estimated_savings"])
        return opportunities

    # ---- Reporting ----

    def summary(self) -> str:
        """Human-readable summary string."""
        lines: List[str] = []
        title = f"Ledger Summary" + (f" ({self.name})" if self.name else "")
        lines.append(title)
        lines.append("=" * len(title))

        budget_str = f" / ${self.budget:.4f}" if self.budget is not None else ""
        lines.append(f"Total spent:      ${self.total_cost:.4f}{budget_str}")
        lines.append(
            f"Total calls:      {self.total_calls}  "
            f"({self.successful_calls} ok, {self.failed_calls} failed)"
        )
        lines.append(
            f"Total tokens:     {self.total_input_tokens:,} in / "
            f"{self.total_output_tokens:,} out"
        )

        # By provider
        by_provider: Dict[str, float] = {}
        for c in self.calls:
            by_provider[c.provider] = by_provider.get(c.provider, 0.0) + c.cost
        if by_provider:
            lines.append("")
            lines.append("By provider:")
            for p, cost in sorted(by_provider.items(), key=lambda x: -x[1]):
                lines.append(f"  {p:14s}${cost:.4f}")

        # By tag (per-agent / per-feature attribution)
        by_tag = self.by_tag()
        # Only show this section if there are real tags (not just _untagged)
        if by_tag and not (len(by_tag) == 1 and "_untagged" in by_tag):
            lines.append("")
            lines.append("By tag:")
            calls_per_tag = self.calls_by_tag()
            for t, cost in sorted(by_tag.items(), key=lambda x: -x[1]):
                count = calls_per_tag.get(t, 0)
                lines.append(f"  {t:14s}${cost:.4f}  ({count} calls)")

        # Waste section - now with per-category dollar breakdown
        waste_by_cat = self.wasted_cost_by_category()
        tokens_by_cat = self.wasted_tokens()
        total_waste = sum(waste_by_cat.values())
        if total_waste > 0 or any(tokens_by_cat.values()):
            waste_pct = (
                100 * total_waste / self.total_cost if self.total_cost else 0
            )
            lines.append("")
            lines.append(
                f"Waste:            ${total_waste:.4f} "
                f"({waste_pct:.1f}% of spend)"
            )
            for cat in ("retries", "truncated", "failed"):
                cost = waste_by_cat.get(cat, 0.0)
                toks = tokens_by_cat.get(cat, 0)
                if cost > 0 or toks > 0:
                    lines.append(
                        f"  {cat:14s}${cost:.4f}  ({toks:,} tokens)"
                    )
            # Show top retried prompts if any
            offenders = self.most_retried_prompts(n=3)
            if offenders:
                lines.append("  Top repeated prompts:")
                for o in offenders:
                    lines.append(
                        f"    {o['prompt_hash'][:12]:12s} "
                        f"x{o['calls']}  ${o['cost']:.4f}"
                    )
            # Show error breakdown if any failures
            errors = self.errors_by_type()
            if errors:
                err_str = ", ".join(f"{k}={v}" for k, v in errors.items())
                lines.append(f"  Errors:       {err_str}")

        # Caching section - only show if caching was actually involved
        if self.total_cached_read_tokens > 0 or self.total_cached_write_tokens > 0:
            lines.append("")
            hit_rate = self.cache_hit_rate() * 100
            savings = self.realized_cache_savings()
            lines.append(
                f"Caching:          {hit_rate:.1f}% hit rate, "
                f"saved ${savings:.4f}"
            )
            lines.append(
                f"  Read tokens:  {self.total_cached_read_tokens:,}"
            )
            if self.total_cached_write_tokens > 0:
                lines.append(
                    f"  Write tokens: {self.total_cached_write_tokens:,}"
                )

        # Missed caching opportunities
        if self.total_cost > 0:
            missed = self.missed_caching_opportunities()
            if missed:
                total_missed = sum(m["estimated_savings"] for m in missed)
                lines.append("")
                lines.append(
                    f"Missed caching opportunities: "
                    f"~${total_missed:.4f} potential savings"
                )
                for m in missed[:3]:
                    lines.append(
                        f"  {m['prompt_hash'][:12]:12s} "
                        f"x{m['count']} @ ~{m['avg_input_tokens']:,} tokens "
                        f"could save ${m['estimated_savings']:.4f}"
                    )

        # Cheaper alternatives - only show models that would actually save money
        if self.total_cost > 0:
            alternatives = self.savings_opportunities()
            if alternatives:
                lines.append("")
                lines.append("Cheaper alternatives (same workload):")
                for model, cost, savings_pct in alternatives[:3]:
                    lines.append(
                        f"  {model:42s} ${cost:.4f}  (save {savings_pct:.0f}%)"
                    )

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serializable dict representation."""
        return {
            "name": self.name,
            "budget": self.budget,
            "total_cost": self.total_cost,
            "remaining_budget": self.remaining_budget,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "wasted_cost": self.wasted_cost,
            "calls": [asdict(c) for c in self.calls],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def track_budget(
    budget: Union[str, int, float, None] = None,
    name: Optional[str] = None,
    on_complete: Optional[Callable[[Ledger], None]] = None,
):
    """Decorator form. Wrap a function so its LLM calls run inside a Ledger.

    Example:
        @track_budget("$0.50", on_complete=lambda l: print(l.summary()))
        def my_agent(query):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Ledger(budget=budget, name=name or func.__name__) as ledger:
                result = func(*args, **kwargs)
                if on_complete is not None:
                    on_complete(ledger)
                return result
        return wrapper
    return decorator


def get_current_ledger() -> Optional[Ledger]:
    """Return the currently active Ledger (or None if not inside a context)."""
    return _current_ledger.get()