"""Monkey-patch Anthropic SDK to feed calls into the active Ledger."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

_patched = False
_originals: Dict[str, Any] = {}


def _hash_prompt(messages: Any, system: Any = None) -> Optional[str]:
    """Stable short hash of the prompt for retry detection."""
    if messages is None:
        return None
    try:
        payload = {"messages": messages, "system": system}
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
    except Exception:
        return None


def _extract_usage(response: Any) -> Dict[str, Any]:
    """Pull tokens, model, and stop_reason out of an Anthropic response."""
    out = {
        "input_tokens": 0,
        "output_tokens": 0,
        "model": "unknown",
        "finish_reason": None,
    }
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            out["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
            out["output_tokens"] = getattr(usage, "output_tokens", 0) or 0
        out["model"] = getattr(response, "model", "unknown") or "unknown"
        # Anthropic uses stop_reason; we normalize 'max_tokens' to itself
        out["finish_reason"] = getattr(response, "stop_reason", None)
    except Exception:
        pass
    return out


def patch_anthropic() -> None:
    """Install patches on Messages.create (sync + async). Idempotent."""
    global _patched
    if _patched:
        return
    try:
        from anthropic.resources.messages import Messages, AsyncMessages
    except ImportError:
        return  # anthropic SDK not installed

    from ..ledger import _current_ledger

    _originals["sync"] = Messages.create
    _originals["async"] = AsyncMessages.create

    def _record(ledger, response, kwargs, duration, error=None):
        prompt_hash = _hash_prompt(
            kwargs.get("messages"), kwargs.get("system")
        )
        if error is not None or response is None:
            ledger.record(
                provider="anthropic",
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
            provider="anthropic",
            model=info["model"],
            input_tokens=info["input_tokens"],
            output_tokens=info["output_tokens"],
            duration=duration,
            finish_reason=info["finish_reason"],
            prompt_hash=prompt_hash,
        )

    def sync_create(self, *args, **kwargs):
        ledger = _current_ledger.get()
        if ledger is None:
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
        if ledger is None:
            return await _originals["async"](self, *args, **kwargs)
        start = time.time()
        try:
            response = await _originals["async"](self, *args, **kwargs)
        except Exception as e:
            _record(ledger, None, kwargs, time.time() - start, error=str(e))
            raise
        _record(ledger, response, kwargs, time.time() - start)
        return response

    Messages.create = sync_create
    AsyncMessages.create = async_create
    _patched = True


def unpatch_anthropic() -> None:
    """Restore the original Anthropic SDK methods. Mainly for testing."""
    global _patched
    if not _patched:
        return
    try:
        from anthropic.resources.messages import Messages, AsyncMessages
        Messages.create = _originals["sync"]
        AsyncMessages.create = _originals["async"]
    except ImportError:
        pass
    _patched = False
