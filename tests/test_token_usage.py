"""Tests for token-usage extraction from provider outputs."""
from concurrent.futures import ThreadPoolExecutor

from explorers.parsing import (
    TokenUsage,
    extract_usage,
    extract_usage_from_jsonl,
    report_usage,
    usage_collector,
)


def test_extract_anthropic_style_usage():
    data = {
        "type": "result",
        "result": "RELEVANT_FILES:\n- a.py:1-10",
        "usage": {
            "input_tokens": 100,
            "cache_creation_input_tokens": 40,
            "cache_read_input_tokens": 900,
            "output_tokens": 55,
            "server_tool_use": {"web_search_requests": 0},
        },
    }
    u = extract_usage(data)
    assert u.input_tokens == 100
    assert u.cache_write_tokens == 40
    assert u.cache_read_tokens == 900
    assert u.output_tokens == 55
    assert u.reasoning_tokens == 0
    assert u.total == 155
    assert u.separate_reasoning_tokens == 0


def test_extract_model_usage_camel_case():
    data = {
        "modelUsage": {
            "claude-sonnet": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadTokens": 7,
                    "cacheCreationTokens": 3,
                },
                "count": 2,
            },
            "claude-haiku": {"usage": {"inputTokens": 1, "outputTokens": 2}},
        }
    }
    u = extract_usage(data)
    assert u.input_tokens == 11
    assert u.output_tokens == 7
    assert u.cache_read_tokens == 7
    assert u.cache_write_tokens == 3


def test_extract_openai_style_subtracts_cached_from_prompt():
    data = {
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 60},
            "completion_tokens_details": {"reasoning_tokens": 8},
        }
    }
    u = extract_usage(data)
    assert u.input_tokens == 40
    assert u.cache_read_tokens == 60
    assert u.output_tokens == 20
    assert u.reasoning_tokens == 8
    assert u.separate_reasoning_tokens == 0
    assert u.total == 60
    assert u.to_dict()["total"] == 60


def test_extract_responses_api_style_subtracts_cached_from_input():
    """Responses API input_tokens already includes the cached portion."""
    data = {
        "usage": {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 60},
            "output_tokens": 20,
        }
    }
    u = extract_usage(data)
    assert u.input_tokens == 40
    assert u.cache_read_tokens == 60
    assert u.output_tokens == 20
    assert u.total == 60


def test_extract_gemini_style_reasoning_is_separate():
    data = {
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 8,
            "cachedContentTokenCount": 60,
        }
    }
    u = extract_usage(data)
    assert u.input_tokens == 40
    assert u.cache_read_tokens == 60
    assert u.output_tokens == 20
    assert u.reasoning_tokens == 8
    assert u.separate_reasoning_tokens == 8
    assert u.total == 40 + 20 + 8


def test_extract_reasoning_without_output_is_separate():
    data = {"usage": {"input_tokens": 10, "reasoning_tokens": 5}}
    u = extract_usage(data)
    assert u.reasoning_tokens == 5
    assert u.separate_reasoning_tokens == 5
    assert u.total == 15


def test_text_reasoning_field_keeps_details_inclusive():
    data = {
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "reasoning": "free-form text transcript",
            "completion_tokens_details": {"reasoning_tokens": 25},
        }
    }
    u = extract_usage(data)
    assert u.output_tokens == 30
    assert u.reasoning_tokens == 25
    assert u.separate_reasoning_tokens == 0
    assert u.total == 80


def test_bare_candidates_count_is_not_output():
    data = {"response": {"candidates": 3}, "usage": {"output_tokens": 7}}
    u = extract_usage(data)
    assert u.output_tokens == 7
    assert u.separate_reasoning_tokens == 0


def test_extract_ignores_text_and_bools():
    data = {"token_usage": {"input_tokens": 5}, "reasoning": "long text", "cached": True}
    u = extract_usage(data)
    assert u.input_tokens == 5
    assert u.reasoning_tokens == 0


def test_extract_usage_from_jsonl_events():
    raw = "\n".join(
        [
            '{"type":"step_start","part":{"id":1}}',
            '{"type":"step_finish","tokens":{"input":3,"output":4,"reasoning":2}}',
            "not json",
            '{"type":"step_finish","tokens":{"input":10,"output":1,"cachedTokens":5}}',
        ]
    )
    u = extract_usage_from_jsonl(raw)
    assert u is not None
    assert u.input_tokens == 13
    assert u.output_tokens == 5
    assert u.reasoning_tokens == 2
    assert u.cache_read_tokens == 5
    # opencode-style tokens objects carry output and reasoning as sibling
    # buckets, so reasoning is not a subset of output.
    assert u.separate_reasoning_tokens == 2
    assert u.total == 20


def test_extract_opencode_style_sibling_reasoning():
    """OpenCode step_finish: reasoning is a sibling bucket of output.

    Real GLM observation: output=4, reasoning=99 in one event — a subset
    cannot exceed its container, so the buckets must be disjoint.
    """
    data = {
        "type": "step_finish",
        "part": {
            "type": "step-finish",
            "tokens": {
                "total": 7346,
                "input": 11,
                "output": 4,
                "reasoning": 99,
                "cache": {"write": 0, "read": 7232},
            },
        },
    }
    u = extract_usage(data)
    assert u.input_tokens == 11
    assert u.output_tokens == 4
    assert u.reasoning_tokens == 99
    assert u.cache_read_tokens == 7232
    assert u.separate_reasoning_tokens == 99
    # Cache is tracked separately and never part of total.
    assert u.total == 11 + 4 + 99


def test_extract_openai_nested_reasoning_stays_inclusive():
    """OpenAI completion_tokens_details keeps reasoning inside output."""
    data = {
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "completion_tokens_details": {"reasoning_tokens": 25},
        }
    }
    u = extract_usage(data)
    assert u.reasoning_tokens == 25
    assert u.separate_reasoning_tokens == 0
    assert u.total == 80


def test_from_dict_legacy_reasoning_flag():
    legacy = {
        "input": 5,
        "output": 4,
        "reasoning": 2,
        "reasoning_separate": True,
        "total": 11,
    }
    u = TokenUsage.from_dict(legacy)
    assert u.separate_reasoning_tokens == 2
    assert u.total == 11


def test_extract_usage_from_jsonl_empty():
    assert extract_usage_from_jsonl("") is None
    assert extract_usage_from_jsonl("hello\nworld") is None


def test_token_usage_roundtrip_and_add():
    a = TokenUsage(input_tokens=10, output_tokens=2, reasoning_tokens=3)
    assert a.total == 12
    assert TokenUsage(cache_read_tokens=99, cache_write_tokens=99).total == 0
    b = TokenUsage.from_dict(a.to_dict())
    assert b == a
    assert b.total == 12

    a.add(b)
    assert a.input_tokens == 20
    assert a.output_tokens == 4
    assert a.reasoning_tokens == 6
    assert a.total == 24
    assert not TokenUsage().has_any()
    assert a.has_any()
    a.add(None)
    assert a.input_tokens == 20


def test_add_mixed_reasoning_provenance():
    included = TokenUsage(input_tokens=10, output_tokens=20, reasoning_tokens=5)
    separate = TokenUsage(
        input_tokens=1,
        output_tokens=2,
        reasoning_tokens=3,
        separate_reasoning_tokens=3,
    )
    assert included.total == 30
    assert separate.total == 6

    total = TokenUsage()
    total.add(included)
    total.add(separate)
    # Aggregation must be additive: no reasoning token counted twice.
    assert total.total == included.total + separate.total == 36
    assert total.reasoning_tokens == 8
    assert total.separate_reasoning_tokens == 3


def test_report_usage_collector_accumulates_per_thread():
    def work(n):
        with usage_collector() as tracker:
            report_usage(TokenUsage(input_tokens=n))
            report_usage(TokenUsage(output_tokens=1))
            report_usage(None)
        return tracker

    with ThreadPoolExecutor(max_workers=4) as pool:
        trackers = list(pool.map(work, range(1, 5)))

    assert [t.input_tokens for t in trackers] == [1, 2, 3, 4]
    assert all(t.output_tokens == 1 for t in trackers)

    report_usage(TokenUsage(input_tokens=999))
