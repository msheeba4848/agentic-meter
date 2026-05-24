"""The Ledger - core tracking, budget enforcement, and reporting."""
from __future__ import annotations

import contextvars
import functools
import json
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
    input_tokens: int
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
    ) -> Call:
        """Manually record a call. Use this for providers without auto-tracking."""
        cost = (
            0.0
            if failed
            else calculate_cost(provider, model, input_tokens, output_tokens)
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

        # Waste section
        retries = self.detect_retries()
        truncs = self.detect_truncations()
        failed = self.detect_failed()
        if retries or truncs or failed:
            waste_pct = (
                100 * self.wasted_cost / self.total_cost if self.total_cost else 0
            )
            lines.append("")
            lines.append(
                f"Waste:            ${self.wasted_cost:.4f} "
                f"({waste_pct:.1f}% of spend)"
            )
            if retries:
                lines.append(
                    f"  Retries:        {len(retries)} calls, "
                    f"${sum(c.cost for c in retries):.4f}"
                )
            if truncs:
                lines.append(
                    f"  Truncated:      {len(truncs)} calls, "
                    f"${sum(c.cost for c in truncs):.4f}"
                )
            if failed:
                lines.append(f"  Failed:         {len(failed)} calls")

        # Counterfactuals - top 3 cheapest alternatives
        if self.total_cost > 0:
            cf = self.counterfactuals()
            cf_sorted = sorted(
                [(k, v) for k, v in cf.items() if v > 0],
                key=lambda x: x[1],
            )
            if cf_sorted:
                lines.append("")
                lines.append("Counterfactual (same workload on alternative models):")
                for model, cost in cf_sorted[:3]:
                    delta = (cost - self.total_cost) / self.total_cost * 100
                    arrow = "down" if delta < 0 else "up"
                    lines.append(
                        f"  {model:42s} ${cost:.4f}  "
                        f"({arrow} {abs(delta):.0f}%)"
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