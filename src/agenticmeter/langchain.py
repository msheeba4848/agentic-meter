"""LangChain callback handler for agenticmeter.

LangChain wraps provider responses in its own `LLMResult` object, and token
usage lives in `llm_output['token_usage']` or `generations[0][0].message.usage_metadata`
rather than `response.usage` like the raw SDK. This module handles that wrapping
correctly and works for both chat models and completion models, sync and async,
streaming and non-streaming.

Use this whenever you're working through LangChain — the raw-SDK monkey-patch
in agenticmeter.trackers will fire but won't extract tokens correctly because
LangChain's calling pattern hides the usage object.

Quickstart:

    from agenticmeter import Ledger
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini")

    with Ledger(budget="$1.00") as ledger:
        cb = ledger.as_langchain_callback()
        response = llm.invoke("hello", config={"callbacks": [cb]})

    print(ledger.summary())
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Stub so the module imports cleanly without langchain installed."""

    class LLMResult:  # type: ignore[no-redef]
        pass


from .ledger import _current_ledger


class agenticmeterCallback(BaseCallbackHandler):
    """LangChain callback that records LLM calls into the active Ledger.

    Created via ``ledger.as_langchain_callback()``. The callback reads the
    currently active Ledger via ContextVar at each event, so a single callback
    instance works across nested contexts.
    """

    def __init__(self):
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain-core is required for agenticmeterCallback. "
                "Install with: pip install 'agenticmeter[langchain]' "
                "or: pip install langchain-core"
            )
        # Per-run state keyed by run_id (UUID)
        self._starts: Dict[UUID, float] = {}
        self._prompts: Dict[UUID, Any] = {}
        self._params: Dict[UUID, Dict[str, Any]] = {}

    # ---- LangChain lifecycle hooks ----

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Chat-model invocation started (ChatOpenAI, ChatAnthropic, etc.)."""
        self._starts[run_id] = time.time()
        # Suppress the raw-SDK monkey-patch for this call. The Ledger attribute
        # is mutated directly (rather than a ContextVar) because LangChain runs
        # callbacks in a copied context where ContextVar.set() wouldn't
        # propagate back to where the SDK call actually fires.
        ledger = _current_ledger.get()
        if ledger is not None:
            with ledger._suppress_lock:
                ledger._suppress_sdk += 1
        try:
            self._prompts[run_id] = [
                [getattr(m, "content", str(m)) for m in msg_list]
                for msg_list in messages
            ]
        except Exception:
            self._prompts[run_id] = None
        self._params[run_id] = kwargs.get("invocation_params") or {}

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        **kwargs: Any,
    ) -> None:
        """Completion-model invocation started (legacy OpenAI completions, etc.)."""
        self._starts[run_id] = time.time()
        ledger = _current_ledger.get()
        if ledger is not None:
            with ledger._suppress_lock:
                ledger._suppress_sdk += 1
        self._prompts[run_id] = prompts
        self._params[run_id] = kwargs.get("invocation_params") or {}

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Invocation completed successfully. Extract usage and record."""
        ledger = _current_ledger.get()
        if ledger is None:
            self._cleanup_run(run_id)
            return

        start = self._starts.get(run_id, time.time())
        duration = time.time() - start
        prompts = self._prompts.get(run_id)
        params = self._params.get(run_id) or {}

        usage, model, provider, finish_reason = self._extract_from_result(
            response, params
        )

        try:
            ledger.record(
                provider=provider,
                model=model,
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                cached_read_tokens=usage.get("cached_read_tokens", 0),
                cached_write_tokens=usage.get("cached_write_tokens", 0),
                duration=duration,
                finish_reason=finish_reason,
                prompt_hash=self._hash_prompt(prompts),
                metadata={
                    "langchain_run_id": str(run_id),
                    "langchain_parent_run_id": (
                        str(parent_run_id) if parent_run_id else None
                    ),
                },
            )
        finally:
            self._cleanup_run(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        ledger = _current_ledger.get()
        if ledger is None:
            self._cleanup_run(run_id)
            return

        start = self._starts.get(run_id, time.time())
        duration = time.time() - start
        prompts = self._prompts.get(run_id)
        params = self._params.get(run_id) or {}
        model = params.get("model") or params.get("model_name") or "unknown"
        provider = self._guess_provider(model)

        try:
            ledger.record(
                provider=provider,
                model=model,
                input_tokens=0,
                output_tokens=0,
                duration=duration,
                failed=True,
                error=str(error),
                prompt_hash=self._hash_prompt(prompts),
                metadata={"langchain_run_id": str(run_id)},
            )
        finally:
            self._cleanup_run(run_id)

    def _cleanup_run(self, run_id: UUID) -> None:
        # Only decrement if we actually started this run (otherwise we'd underflow)
        if run_id in self._starts:
            ledger = _current_ledger.get()
            if ledger is not None:
                with ledger._suppress_lock:
                    if ledger._suppress_sdk > 0:
                        ledger._suppress_sdk -= 1
        self._starts.pop(run_id, None)
        self._prompts.pop(run_id, None)
        self._params.pop(run_id, None)

    # ---- Extraction ----

    @staticmethod
    def _extract_from_result(
        response: LLMResult, params: Dict
    ) -> Tuple[Dict[str, int], str, str, Optional[str]]:
        """Pull tokens/model/finish_reason out of an LLMResult.

        Tries multiple known locations because LangChain providers vary:
          1. response.llm_output['token_usage'] (OpenAI shape)
          2. response.llm_output['usage'] (some Anthropic versions)
          3. response.generations[0][0].message.usage_metadata (newer langchain-openai)
          4. response.generations[0][0].generation_info (finish_reason)
        Cache fields live in input_token_details inside usage_metadata.
        """
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_read_tokens": 0,
            "cached_write_tokens": 0,
        }
        model = params.get("model") or params.get("model_name") or "unknown"
        finish_reason: Optional[str] = None

        llm_output = getattr(response, "llm_output", None) or {}

        # Path 1: llm_output['token_usage'] (most common, OpenAI-style)
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if isinstance(token_usage, dict):
            # OpenAI naming
            if "prompt_tokens" in token_usage:
                raw_prompt = int(token_usage.get("prompt_tokens", 0) or 0)
                cached = 0
                # OpenAI nests cached_tokens in prompt_tokens_details
                details = token_usage.get("prompt_tokens_details") or {}
                if isinstance(details, dict):
                    cached = int(details.get("cached_tokens", 0) or 0)
                usage["cached_read_tokens"] = cached
                usage["input_tokens"] = max(0, raw_prompt - cached)
                usage["output_tokens"] = int(
                    token_usage.get("completion_tokens", 0) or 0
                )
            # Anthropic naming
            elif "input_tokens" in token_usage:
                usage["input_tokens"] = int(token_usage.get("input_tokens", 0) or 0)
                usage["output_tokens"] = int(
                    token_usage.get("output_tokens", 0) or 0
                )
                usage["cached_read_tokens"] = int(
                    token_usage.get("cache_read_input_tokens", 0) or 0
                )
                usage["cached_write_tokens"] = int(
                    token_usage.get("cache_creation_input_tokens", 0) or 0
                )

        # Override model name if llm_output has a more accurate one
        if llm_output.get("model_name"):
            model = llm_output["model_name"]
        elif llm_output.get("model"):
            model = llm_output["model"]

        # Path 2 & finish_reason: dig into generations
        try:
            generations = getattr(response, "generations", None) or []
            if generations and generations[0]:
                gen = generations[0][0]
                gen_info = getattr(gen, "generation_info", None) or {}
                finish_reason = gen_info.get("finish_reason") or gen_info.get(
                    "stop_reason"
                )

                # Fallback: pull from the message's usage_metadata if still 0
                if usage["input_tokens"] == 0 and usage["output_tokens"] == 0:
                    message = getattr(gen, "message", None)
                    um = getattr(message, "usage_metadata", None) if message else None
                    if isinstance(um, dict):
                        in_tok = int(um.get("input_tokens", 0) or 0)
                        usage["output_tokens"] = int(um.get("output_tokens", 0) or 0)
                        # Cache details live in input_token_details
                        in_details = um.get("input_token_details") or {}
                        if isinstance(in_details, dict):
                            usage["cached_read_tokens"] = int(
                                in_details.get("cache_read", 0) or 0
                            )
                            usage["cached_write_tokens"] = int(
                                in_details.get("cache_creation", 0) or 0
                            )
                        # For OpenAI-style: in_tok includes cached; subtract.
                        # For Anthropic-style: in_tok is already uncached.
                        # We can't easily tell here, so we infer from cache_creation:
                        # if cache_creation > 0, it's Anthropic-style.
                        if usage["cached_write_tokens"] > 0:
                            usage["input_tokens"] = in_tok
                        else:
                            usage["input_tokens"] = max(
                                0, in_tok - usage["cached_read_tokens"]
                            )

                # Fallback model name from message.response_metadata
                if model == "unknown":
                    message = getattr(gen, "message", None)
                    rm = (
                        getattr(message, "response_metadata", None) if message else None
                    )
                    if isinstance(rm, dict):
                        model = rm.get("model_name") or rm.get("model") or model
        except (AttributeError, IndexError, TypeError):
            pass

        provider = agenticmeterCallback._guess_provider(model)
        return usage, model, provider, finish_reason

    @staticmethod
    def _guess_provider(model: str) -> str:
        """Best-effort provider inference from model name."""
        if not model:
            return "unknown"
        m = model.lower()
        if "claude" in m:
            return "anthropic"
        if "gpt" in m or m.startswith(("o1", "o3", "o4")) or "davinci" in m:
            return "openai"
        if "gemini" in m or "palm" in m or "bison" in m:
            return "google"
        if "mistral" in m or "mixtral" in m:
            return "mistral"
        return "unknown"

    @staticmethod
    def _hash_prompt(prompts: Any) -> Optional[str]:
        if prompts is None:
            return None
        try:
            s = json.dumps(prompts, sort_keys=True, default=str)
            return hashlib.sha256(s.encode()).hexdigest()[:16]
        except Exception:
            return None

        
