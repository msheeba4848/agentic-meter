"""Monkey-patch boto3 Bedrock Runtime clients to feed calls into the active Ledger.

Bedrock is a router: one API surface, many underlying model families with different
response shapes. This tracker handles the three most common ones:

- Anthropic Claude on Bedrock: response body has usage.input_tokens / usage.output_tokens
  (same shape as the direct Anthropic API, including cache_read/creation tokens)
- Meta Llama on Bedrock: response body has prompt_token_count / generation_token_count
- Amazon Titan on Bedrock: response body has inputTextTokenCount / results[].tokenCount

For unknown model families, the tracker records the call with 0 tokens rather than
crashing. Cost will be $0 until you add the model to prices.json.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

_patched = False
_originals: Dict[str, Any] = {}


def _hash_body(body: Any) -> Optional[str]:
    """Stable short hash of the request body for retry detection."""
    if body is None:
        return None
    try:
        # body is usually a JSON string or dict
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8", errors="replace")
        if isinstance(body, str):
            # Normalize by parsing then re-serializing sorted
            try:
                parsed = json.loads(body)
                s = json.dumps(parsed, sort_keys=True, default=str)
            except (json.JSONDecodeError, ValueError):
                s = body
        else:
            s = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()[:16]
    except Exception:
        return None


def _infer_provider_from_model_id(model_id: str) -> str:
    """Bedrock model IDs are 'anthropic.claude-...', 'meta.llama-...', etc."""
    if not model_id:
        return "bedrock"
    prefix = model_id.split(".", 1)[0].lower()
    if prefix == "anthropic":
        return "anthropic"
    if prefix in ("meta", "cohere", "ai21", "amazon", "mistral", "stability"):
        return prefix
    return "bedrock"


def _normalize_model_id(model_id: str) -> str:
    """Strip Bedrock region prefixes and the 'anthropic.' vendor prefix.

    Bedrock model IDs come in shapes like:
      - anthropic.claude-3-5-sonnet-20241022-v2:0
      - us.anthropic.claude-3-5-sonnet-20241022-v2:0  (inference profile)
      - meta.llama3-70b-instruct-v1:0

    We strip regional prefix (us./eu./etc.) and the vendor namespace so that
    prices.json can match on the model name proper. We keep the version suffix.
    """
    if not model_id:
        return "unknown"
    # Strip inference profile region prefix
    parts = model_id.split(".")
    if len(parts) >= 3 and len(parts[0]) <= 3:
        # e.g. 'us', 'eu', 'apac' - regional profile
        parts = parts[1:]
    if len(parts) >= 2:
        # 'anthropic.claude-3-5-sonnet-...' -> 'claude-3-5-sonnet-...'
        parts = parts[1:]
    return ".".join(parts) if parts else model_id


def _extract_usage_claude(body: Dict) -> Dict[str, int]:
    """Anthropic Claude on Bedrock returns Anthropic-style usage."""
    usage = body.get("usage", {}) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cached_read_tokens": int(
            usage.get("cache_read_input_tokens", 0) or 0
        ),
        "cached_write_tokens": int(
            usage.get("cache_creation_input_tokens", 0) or 0
        ),
        "finish_reason": body.get("stop_reason"),
    }


def _extract_usage_llama(body: Dict) -> Dict[str, int]:
    """Meta Llama on Bedrock uses different field names."""
    return {
        "input_tokens": int(body.get("prompt_token_count", 0) or 0),
        "output_tokens": int(body.get("generation_token_count", 0) or 0),
        "cached_read_tokens": 0,
        "cached_write_tokens": 0,
        "finish_reason": body.get("stop_reason"),
    }


def _extract_usage_titan(body: Dict) -> Dict[str, int]:
    """Amazon Titan on Bedrock: input token count at top level, output on each result."""
    input_tokens = int(body.get("inputTextTokenCount", 0) or 0)
    output_tokens = 0
    finish_reason = None
    results = body.get("results", []) or []
    if results:
        first = results[0] or {}
        output_tokens = int(first.get("tokenCount", 0) or 0)
        finish_reason = first.get("completionReason")
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_read_tokens": 0,
        "cached_write_tokens": 0,
        "finish_reason": finish_reason,
    }


def _extract_usage_converse(response: Dict) -> Dict[str, Any]:
    """The Converse API is normalized across model families.

    Response has top-level 'usage' with inputTokens / outputTokens, and
    optionally cacheReadInputTokens / cacheWriteInputTokens for Claude models
    with prompt caching enabled.
    """
    usage = response.get("usage", {}) or {}
    return {
        "input_tokens": int(usage.get("inputTokens", 0) or 0),
        "output_tokens": int(usage.get("outputTokens", 0) or 0),
        "cached_read_tokens": int(usage.get("cacheReadInputTokens", 0) or 0),
        "cached_write_tokens": int(usage.get("cacheWriteInputTokens", 0) or 0),
        "finish_reason": response.get("stopReason"),
    }


def _parse_invoke_response(response: Any, model_id: str) -> Dict[str, Any]:
    """Read the body stream from an invoke_model response and extract usage.

    Bedrock's invoke_model returns a StreamingBody in response['body']. We
    consume it once here; downstream consumers who wanted the body will need
    to re-set it. To be non-breaking, we re-inject the parsed body into the
    response dict so downstream code can still call response['body'].read().
    """
    provider = _infer_provider_from_model_id(model_id)
    out = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_read_tokens": 0,
        "cached_write_tokens": 0,
        "finish_reason": None,
    }
    if not isinstance(response, dict):
        return out
    body_stream = response.get("body")
    if body_stream is None:
        return out
    try:
        raw = body_stream.read()
        parsed = json.loads(raw)
    except Exception:
        return out

    # Re-inject a readable body so caller code that does response['body'].read()
    # after our patch still works. Use a small shim.
    class _ReplayBody:
        def __init__(self, raw_bytes):
            self._raw = raw_bytes
            self._consumed = False

        def read(self, *args, **kwargs):
            if self._consumed:
                return b""
            self._consumed = True
            return self._raw

        def close(self):
            pass

    response["body"] = _ReplayBody(raw)

    # Route to the right extractor based on model family
    if provider == "anthropic":
        out.update(_extract_usage_claude(parsed))
    elif provider == "meta":
        out.update(_extract_usage_llama(parsed))
    elif provider == "amazon":
        out.update(_extract_usage_titan(parsed))
    else:
        # Unknown family: try Claude shape first (most common), then Titan
        if "usage" in parsed:
            out.update(_extract_usage_claude(parsed))
        elif "inputTextTokenCount" in parsed:
            out.update(_extract_usage_titan(parsed))
        elif "prompt_token_count" in parsed:
            out.update(_extract_usage_llama(parsed))
    return out


def patch_bedrock() -> None:
    """Install patches on boto3 bedrock-runtime client methods. Idempotent.

    boto3 clients are dynamically generated, so we patch the ClientCreator
    to install our wrappers on any bedrock-runtime client at creation time.
    """
    global _patched
    if _patched:
        return
    try:
        import botocore.client
    except ImportError:
        return  # boto3/botocore not installed

    from ..ledger import _current_ledger

    _originals["_create_client"] = botocore.client.ClientCreator.create_client

    def _record(ledger, provider, model_id, response, body, duration, error=None):
        prompt_hash = _hash_body(body)
        # Normalize model id for prices.json lookup
        normalized_model = _normalize_model_id(model_id) if model_id else "unknown"

        if error is not None or response is None:
            ledger.record(
                provider=provider,
                model=normalized_model,
                input_tokens=0,
                output_tokens=0,
                duration=duration,
                failed=True,
                error=error,
                prompt_hash=prompt_hash,
                metadata={"bedrock_model_id": model_id},
            )
            return

        # Detect API: invoke_model vs converse
        if "output" in response and "message" in (response.get("output") or {}):
            # Converse API response
            info = _extract_usage_converse(response)
        else:
            # invoke_model response
            info = _parse_invoke_response(response, model_id)

        ledger.record(
            provider=provider,
            model=normalized_model,
            input_tokens=info["input_tokens"],
            output_tokens=info["output_tokens"],
            cached_read_tokens=info["cached_read_tokens"],
            cached_write_tokens=info["cached_write_tokens"],
            duration=duration,
            finish_reason=info.get("finish_reason"),
            prompt_hash=prompt_hash,
            metadata={"bedrock_model_id": model_id},
        )

    def _wrap_invoke_model(original):
        def wrapped(self, *args, **kwargs):
            ledger = _current_ledger.get()
            if ledger is None or ledger._suppress_sdk > 0:
                return original(*args, **kwargs)
            model_id = kwargs.get("modelId") or ""
            body = kwargs.get("body")
            provider = _infer_provider_from_model_id(model_id)
            start = time.time()
            try:
                response = original(*args, **kwargs)
            except Exception as e:
                _record(
                    ledger, provider, model_id, None, body,
                    time.time() - start, error=str(e),
                )
                raise
            _record(
                ledger, provider, model_id, response, body,
                time.time() - start,
            )
            return response
        return wrapped

    def _wrap_converse(original):
        def wrapped(self, *args, **kwargs):
            ledger = _current_ledger.get()
            if ledger is None or ledger._suppress_sdk > 0:
                return original(*args, **kwargs)
            model_id = kwargs.get("modelId") or ""
            body = kwargs.get("messages")
            provider = _infer_provider_from_model_id(model_id)
            start = time.time()
            try:
                response = original(*args, **kwargs)
            except Exception as e:
                _record(
                    ledger, provider, model_id, None, body,
                    time.time() - start, error=str(e),
                )
                raise
            _record(
                ledger, provider, model_id, response, body,
                time.time() - start,
            )
            return response
        return wrapped

    original_create = _originals["_create_client"]

    def patched_create_client(self, service_name, *args, **kwargs):
        client = original_create(self, service_name, *args, **kwargs)
        if service_name == "bedrock-runtime":
            # Wrap invoke_model and converse if present
            if hasattr(client, "invoke_model") and not getattr(
                client.invoke_model, "_agenticmeter_wrapped", False
            ):
                orig_invoke = client.invoke_model
                wrapper = _wrap_invoke_model(orig_invoke)
                wrapper._agenticmeter_wrapped = True
                client.invoke_model = wrapper.__get__(client, type(client))
            if hasattr(client, "converse") and not getattr(
                client.converse, "_agenticmeter_wrapped", False
            ):
                orig_converse = client.converse
                wrapper = _wrap_converse(orig_converse)
                wrapper._agenticmeter_wrapped = True
                client.converse = wrapper.__get__(client, type(client))
        return client

    botocore.client.ClientCreator.create_client = patched_create_client
    _patched = True


def unpatch_bedrock() -> None:
    """Restore the original boto3 client creator. Mainly for testing."""
    global _patched
    if not _patched:
        return
    try:
        import botocore.client
        botocore.client.ClientCreator.create_client = _originals["_create_client"]
    except ImportError:
        pass
    _patched = False
