"""OAuth request shape required for included subscription usage."""

from __future__ import annotations

import pytest

from agent.anthropic_adapter import (
    _CLAUDE_CODE_SYSTEM_PREFIX,
    build_anthropic_kwargs,
)
from agent.prompt_caching import apply_anthropic_cache_control


def _build(messages, *, is_oauth=True):
    return build_anthropic_kwargs(
        model="claude-sonnet-4-6",
        messages=messages,
        tools=None,
        max_tokens=32,
        reasoning_config=None,
        is_oauth=is_oauth,
    )


def _first_user(kwargs):
    return next(msg for msg in kwargs["messages"] if msg["role"] == "user")


@pytest.mark.parametrize(
    ("ttl", "expected_marker"),
    [
        ("5m", {"type": "ephemeral"}),
        ("1h", {"type": "ephemeral", "ttl": "1h"}),
    ],
)
def test_oauth_relocates_full_system_prompt_and_preserves_cache_ttl(
    ttl,
    expected_marker,
):
    prompt = "You are Hermes Agent built by Nous Research.\n" + "RULE\n" * 2_000
    cached = apply_anthropic_cache_control(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "do the thing"},
        ],
        cache_ttl=ttl,
        native_anthropic=True,
    )

    kwargs = _build(cached)

    assert kwargs["system"] == [
        {"type": "text", "text": _CLAUDE_CODE_SYSTEM_PREFIX}
    ]
    blocks = _first_user(kwargs)["content"]
    relocated = blocks[0]
    assert relocated["cache_control"] == expected_marker
    assert relocated["text"].startswith("<system_context>\n")
    assert relocated["text"].endswith("\n</system_context>")
    assert relocated["text"].count("RULE\n") == 2_000
    assert "Hermes Agent" not in relocated["text"]
    assert "Nous Research" not in relocated["text"]
    assert any(block.get("text") == "do the thing" for block in blocks)


def test_oauth_does_not_enable_caching_when_disabled():
    kwargs = _build(
        [
            {"role": "system", "content": "Hermes Agent system prompt"},
            {"role": "user", "content": "hello"},
        ]
    )

    relocated = _first_user(kwargs)["content"][0]
    assert "cache_control" not in relocated


def test_oauth_relocation_stays_within_four_cache_breakpoints():
    cached = apply_anthropic_cache_control(
        [
            {"role": "system", "content": "Hermes Agent prompt"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "second"},
        ],
        cache_ttl="1h",
        native_anthropic=True,
    )

    kwargs = _build(cached)

    def count_markers(value):
        if isinstance(value, dict):
            return int("cache_control" in value) + sum(
                count_markers(item) for item in value.values()
            )
        if isinstance(value, list):
            return sum(count_markers(item) for item in value)
        return 0

    assert count_markers(kwargs) <= 4


def test_non_oauth_keeps_system_prompt_out_of_user_content():
    kwargs = _build(
        [
            {"role": "system", "content": "Hermes Agent system prompt"},
            {"role": "user", "content": "hello"},
        ],
        is_oauth=False,
    )

    assert kwargs["system"] == "Hermes Agent system prompt"
    assert "system_context" not in str(_first_user(kwargs)["content"])
