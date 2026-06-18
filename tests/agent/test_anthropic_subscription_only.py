"""Tests for native Anthropic subscription-only auth mode."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from agent import anthropic_adapter as aa
from hermes_cli import runtime_provider as rp


def _write_claude_code_credentials(tmp_path, *, scopes=None, refresh_token="refresh-token"):
    cred_path = tmp_path / ".claude" / ".credentials.json"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oauth-file-token",
                    "refreshToken": refresh_token,
                    "expiresAt": int(time.time() * 1000) + 3_600_000,
                    "scopes": scopes if scopes is not None else ["user:inference", "user:profile"],
                }
            }
        ),
        encoding="utf-8",
    )
    return cred_path


def test_subscription_only_ignores_env_and_explicit_api_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-env")
    monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat-env")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "cc-env")
    _write_claude_code_credentials(tmp_path)

    creds = aa.resolve_anthropic_credentials(
        auth_mode="subscription_only",
        explicit_api_key="sk-ant-api-explicit",
    )

    assert creds.token == "sk-ant-oauth-file-token"
    assert creds.auth_mode == "subscription_only"
    assert creds.source == "claude_code_credentials_file"
    assert creds.is_oauth is True
    assert set(creds.ignored_sources) == {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
    }


def test_subscription_only_fails_without_claude_code_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-env")

    with pytest.raises(aa.AnthropicAuthError, match="requires Claude Code OAuth"):
        aa.resolve_anthropic_credentials(auth_mode="subscription_only")


def test_subscription_only_requires_inference_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_claude_code_credentials(tmp_path, scopes=["user:profile"])

    with pytest.raises(aa.AnthropicAuthError, match="user:inference"):
        aa.resolve_anthropic_credentials(auth_mode="subscription_only")


def test_runtime_provider_subscription_only_bypasses_anthropic_pool(monkeypatch):
    monkeypatch.setattr(rp, "_get_model_config", lambda: {
        "provider": "anthropic",
        "auth_mode": "subscription_only",
    })
    monkeypatch.setattr(rp, "resolve_provider", lambda *a, **k: "anthropic")
    monkeypatch.setattr(rp, "_resolve_named_custom_runtime", lambda **k: None)
    monkeypatch.setattr(rp, "load_pool", lambda provider: pytest.fail("pool must not be loaded"))
    monkeypatch.setattr(
        aa,
        "resolve_anthropic_credentials",
        lambda *, auth_mode=None, explicit_api_key=None: SimpleNamespace(
            token="sk-ant-oauth-file-token",
            source="claude_code_credentials_file",
            auth_mode="subscription_only",
            ignored_sources=("ANTHROPIC_API_KEY",),
        ),
    )

    resolved = rp.resolve_runtime_provider(requested="anthropic")

    assert resolved["provider"] == "anthropic"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["api_key"] == "sk-ant-oauth-file-token"
    assert resolved["auth_mode"] == "subscription_only"
    assert resolved["auth_source"] == "claude_code_credentials_file"
