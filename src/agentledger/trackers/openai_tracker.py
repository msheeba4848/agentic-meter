"""Monkey-patch OpenAI SDK to feed calls into the active Ledger."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

_patched = False
_originals: Dict[str, Any] = {}


def _hash_prompt(messages: Any) -> Optional[str]:
    """Stable short hash of the prompt for retry detection."""
    if messages is None:
        return None
    try:
        s = json.dumps(messages, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
    except Exception:
        return None


def _extract_usage(response: Any) -> Dict[str, Any]:
    """Pull tokens, model, and finish_reason out of an OpenAI response.

    OpenAI's prompt_tokens INCLUDES the cached portion, so we subtract to get
    the uncached input. Cached tokens come from prompt_tokens_details.cached_tokens.
    """
    out = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_read_tokens": 0,
        "model": "unknown",
        "finish_reason": None,
    }
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            raw_prompt = getattr(usage, "prompt_tokens", 0) or 0
            out["output_tokens"] = getattr(usage, "completion_tokens", 0) or 0
            # Cached tokens are reported in prompt_tokens_details (added 2024)
            details = getattr(usage, "prompt_tokens_details", None)
            cached = 0
            if details is not None:
                cached = getattr(details, "cached_tokens", 0) or 0
            out["cached_read_tokens"] = cached
            # Subtract cached from input so we don't double-bill
            out["input_tokens"] = max(0, raw_prompt - cached)
        out["model"] = getattr(response, "model", "unknown") or "unknown"
        choices = getattr(response, "choices", None) or []
        if choices:
            out["finish_reason"] = getattr(choices[0], "finish_reason", None)
    except Exception:
        pass
    return out


def patch_openai() -> None:
    """Install patches on Completions.create (sync + async). Idempotent."""
    global _patched
    if _patched:
        return
    try:
        from openai.resources.chat.completions import (
            Completions,
            AsyncCompletions,
        )
    except ImportError:
        return  # openai SDK not installed

    # Late import to avoid circular dependency
    from ..ledger import _current_ledger

    _originals["sync"] = Completions.create
    _originals["async"] = AsyncCompletions.create

    def _record(ledger, response, kwargs, duration, error=None):
        prompt_hash = _hash_prompt(kwargs.get("messages"))
        if error is not None or response is None:
            ledger.record(
                provider="openai",
                model=kwargs.get("model", "unknown"),
                input_tokens=0,
                output_tokens=0,
                duration=duration,
                failed=True,
                error=error,
                prompt_hash=prompt_hash,
            )
            return
        info = _extract_usage(response)
        ledger.record(
            provider="openai",
            model=info["model"],
            input_tokens=info["input_tokens"],
            output_tokens=info["output_tokens"],
            cached_read_tokens=info["cached_read_tokens"],
            duration=duration,
            finish_reason=info["finish_reason"],
            prompt_hash=prompt_hash,
        )

    def sync_create(self, *args, **kwargs):
        ledger = _current_ledger.get()
        # Bail if no ledger active, or if a higher-level integration (like the
        # LangChain callback) is already handling this call - the suppression
        # attribute is on the Ledger instance so it's visible across contexts.
        if ledger is None or ledger._suppress_sdk > 0:
            return _originals["sync"](self, *args, **kwargs)
        start = time.time()
        try:
            response = _originals["sync"](self, *args, **kwargs)
        except Exception as e:
            _record(ledger, None, kwargs, time.time() - start, error=str(e))
            raise
        _record(ledger, response, kwargs, time.time() - start)
        return response

    async def async_create(self, *args, **kwargs):
        ledger = _current_ledger.get()
        if ledger is None or ledger._suppress_sdk > 0:
            return await _originals["async"](self, *args, **kwargs)
        start = time.time()
        try:
            response = await _originals["async"](self, *args, **kwargs)
        except Exception as e:
            _record(ledger, None, kwargs, time.time() - start, error=str(e))
            raise
        _record(ledger, response, kwargs, time.time() - start)
        return response

    Completions.create = sync_create
    AsyncCompletions.create = async_create
    _patched = True


def unpatch_openai() -> None:
    """Restore the original OpenAI SDK methods. Mainly for testing."""
    global _patched
    if not _patched:
        return
    try:
        from openai.resources.chat.completions import (
            Completions,
            AsyncCompletions,
        )
        Completions.create = _originals["sync"]
        AsyncCompletions.create = _originals["async"]
    except ImportError:
        pass
    _patched = False