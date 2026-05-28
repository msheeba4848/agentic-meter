"""Tests for the LangChain callback handler.

These tests use mock LLMResult objects so they don't require LangChain to be
installed at test time - they verify the extraction logic against the data
shapes LangChain produces.
"""
import pytest
from uuid import uuid4
from agenticmeter import Ledger
from agenticmeter.langchain import agenticmeterCallback, _HAS_LANGCHAIN


# In case the langchain-core isnt installed, all these tests are skipped.

pytestmark = pytest.mark.skipif(
    not _HAS_LANGCHAIN, reason="langchain-core not installed"
)


class MockGeneration:
    """Mimics langchain_core.outputs.ChatGeneration shape."""

    def __init__(self, generation_info=None, message=None):
        self.generation_info = generation_info or {}
        self.message = message


class MockMessage:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata or {}
        self.response_metadata = response_metadata or {}


class MockLLMResult:
    """Mimics langchain_core.outputs.LLMResult shape."""

    def __init__(self, llm_output=None, generations=None):
        self.llm_output = llm_output or {}
        self.generations = generations or []


def test_extract_openai_token_usage_from_llm_output():
    cb = agenticmeterCallback()
    result = MockLLMResult(
        llm_output={
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "model_name": "gpt-4o-mini",
        },
        generations=[[MockGeneration(generation_info={"finish_reason": "stop"})]],
    )
    usage, model, provider, finish_reason = cb._extract_from_result(result, {})
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert model == "gpt-4o-mini"
    assert provider == "openai"
    assert finish_reason == "stop"


def test_extract_anthropic_token_usage_from_llm_output():
    cb = agenticmeterCallback()
    result = MockLLMResult(
        llm_output={
            "usage": {"input_tokens": 200, "output_tokens": 75},
            "model_name": "claude-3-5-sonnet-20241022",
        }
    )
    usage, model, provider, _ = cb._extract_from_result(result, {})
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 75
    assert provider == "anthropic"


def test_extract_falls_back_to_message_usage_metadata():
    """Newer langchain-openai puts usage_metadata on the message, not llm_output."""
    cb = agenticmeterCallback()
    message = MockMessage(
        usage_metadata={"input_tokens": 30, "output_tokens": 15},
        response_metadata={"model_name": "gpt-4o-mini", "finish_reason": "stop"},
    )
    result = MockLLMResult(
        llm_output={},  # empty - the new shape
        generations=[[MockGeneration(message=message)]],
    )
    usage, model, _, _ = cb._extract_from_result(result, {"model_name": "gpt-4o-mini"})
    assert usage["input_tokens"] == 30
    assert usage["output_tokens"] == 15
    assert model == "gpt-4o-mini"


def test_callback_records_into_active_ledger():
    cb = agenticmeterCallback()
    run_id = uuid4()

    result = MockLLMResult(
        llm_output={
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model_name": "gpt-4o-mini",
        },
        generations=[[MockGeneration(generation_info={"finish_reason": "stop"})]],
    )

    with Ledger() as ledger:
        cb.on_chat_model_start(
            serialized={}, messages=[[]], run_id=run_id,
            invocation_params={"model_name": "gpt-4o-mini"},
        )
        cb.on_llm_end(result, run_id=run_id)

    assert ledger.total_calls == 1
    assert ledger.total_input_tokens == 100
    assert ledger.total_output_tokens == 50
    assert ledger.total_cost > 0


def test_callback_records_failure_on_error():
    cb = agenticmeterCallback()
    run_id = uuid4()

    with Ledger() as ledger:
        cb.on_chat_model_start(
            serialized={}, messages=[[]], run_id=run_id,
            invocation_params={"model_name": "gpt-4o"},
        )
        cb.on_llm_error(RuntimeError("rate limited"), run_id=run_id)

    assert ledger.failed_calls == 1
    assert ledger.successful_calls == 0


def test_callback_no_op_when_no_active_ledger():
    """The callback should silently do nothing when no Ledger is active."""
    cb = agenticmeterCallback()
    run_id = uuid4()

    cb.on_chat_model_start(serialized={}, messages=[[]], run_id=run_id)
    cb.on_llm_end(
        MockLLMResult(llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
        run_id=run_id,
    )
    # No error, no ledger to inspect - just shouldn't crash


def test_callback_with_tags_via_ledger():
    """Tagged blocks should attribute callback-recorded calls correctly."""
    cb = agenticmeterCallback()

    result_a = MockLLMResult(
        llm_output={
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "model_name": "gpt-4o-mini",
        }
    )
    result_b = MockLLMResult(
        llm_output={
            "token_usage": {"prompt_tokens": 200, "completion_tokens": 100},
            "model_name": "gpt-4o",
        }
    )

    with Ledger() as ledger:
        with ledger.tag("researcher"):
            rid = uuid4()
            cb.on_chat_model_start(serialized={}, messages=[[]], run_id=rid)
            cb.on_llm_end(result_a, run_id=rid)
        with ledger.tag("writer"):
            rid = uuid4()
            cb.on_chat_model_start(serialized={}, messages=[[]], run_id=rid)
            cb.on_llm_end(result_b, run_id=rid)

    by_tag = ledger.by_tag()
    assert "researcher" in by_tag
    assert "writer" in by_tag
    assert by_tag["writer"] > by_tag["researcher"]  # gpt-4o > gpt-4o-mini


def test_provider_inference():
    cb = agenticmeterCallback()
    assert cb._guess_provider("gpt-4o-mini") == "openai"
    assert cb._guess_provider("claude-3-5-sonnet-20241022") == "anthropic"
    assert cb._guess_provider("o1-preview") == "openai"
    assert cb._guess_provider("gemini-1.5-pro") == "google"
    assert cb._guess_provider("mistral-large-latest") == "mistral"
    assert cb._guess_provider("some-weird-model") == "unknown"