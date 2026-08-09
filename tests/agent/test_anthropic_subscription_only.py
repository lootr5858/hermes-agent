"""Behavioral contract for native Anthropic subscription-only auth."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import anthropic_adapter as aa
from hermes_cli import runtime_provider as rp
from run_agent import AIAgent


def _write_claude_code_credentials(
    tmp_path,
    *,
    scopes=None,
    refresh_token="refresh-token",
):
    cred_path = tmp_path / ".claude" / ".credentials.json"
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-ant-oauth-file-token",
                    "refreshToken": refresh_token,
                    "expiresAt": int(time.time() * 1000) + 3_600_000,
                    "scopes": scopes
                    if scopes is not None
                    else ["user:inference", "user:profile"],
                }
            }
        ),
        encoding="utf-8",
    )


def test_subscription_only_ignores_metered_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(aa.Path, "home", lambda: tmp_path)
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
        "explicit_api_key",
    }


def test_subscription_only_fails_closed_without_refreshable_credentials(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(aa.Path, "home", lambda: tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-env")

    with pytest.raises(aa.AnthropicAuthError, match="requires Claude Code OAuth"):
        aa.resolve_anthropic_credentials(auth_mode="subscription_only")

    _write_claude_code_credentials(tmp_path, refresh_token="")
    with pytest.raises(aa.AnthropicAuthError, match="refresh token"):
        aa.resolve_anthropic_credentials(auth_mode="subscription_only")


def test_subscription_only_requires_inference_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(aa.Path, "home", lambda: tmp_path)
    _write_claude_code_credentials(tmp_path, scopes=["user:profile"])

    with pytest.raises(aa.AnthropicAuthError, match="user:inference"):
        aa.resolve_anthropic_credentials(auth_mode="subscription_only")


def test_default_token_entrypoint_honors_configured_subscription_mode(monkeypatch):
    strict = SimpleNamespace(token="sk-ant-oat01-strict-token")
    monkeypatch.setattr(
        aa,
        "get_configured_anthropic_auth_mode",
        lambda **kwargs: "subscription_only",
    )
    strict_resolver = MagicMock(return_value=strict)
    monkeypatch.setattr(aa, "resolve_anthropic_subscription_credentials", strict_resolver)
    monkeypatch.setattr(
        aa,
        "_resolve_anthropic_token_default",
        lambda: pytest.fail("configured strict mode must not use default resolution"),
    )

    assert aa.resolve_anthropic_token() == "sk-ant-oat01-strict-token"
    strict_resolver.assert_called_once_with()


def test_runtime_provider_subscription_only_bypasses_pool(monkeypatch):
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "anthropic", "auth_mode": "subscription_only"},
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda provider: pytest.fail("strict mode must not inspect the pool"),
    )
    monkeypatch.setattr(
        aa,
        "resolve_anthropic_credentials",
        lambda **kwargs: SimpleNamespace(
            token="sk-ant-oauth-file-token",
            source="claude_code_credentials_file",
            auth_mode="subscription_only",
            ignored_sources=("ANTHROPIC_API_KEY",),
        ),
        raising=False,
    )

    resolved = rp.resolve_runtime_provider(requested="anthropic")

    assert resolved["provider"] == "anthropic"
    assert resolved["api_mode"] == "anthropic_messages"
    assert resolved["api_key"] == "sk-ant-oauth-file-token"
    assert resolved["auth_mode"] == "subscription_only"
    assert resolved["auth_source"] == "claude_code_credentials_file"
    assert resolved["ignored_auth_sources"] == ["ANTHROPIC_API_KEY"]


def test_subscription_only_rejects_non_native_endpoint(monkeypatch):
    monkeypatch.setattr(
        rp,
        "_get_model_config",
        lambda: {"provider": "anthropic", "auth_mode": "subscription_only"},
    )
    monkeypatch.setattr(
        rp,
        "load_pool",
        lambda provider: pytest.fail("strict mode must not inspect the pool"),
    )

    with pytest.raises(rp.AuthError, match="subscription_only.*api.anthropic.com"):
        rp.resolve_runtime_provider(
            requested="anthropic",
            explicit_base_url="https://anthropic.example.net/v1",
        )


def _make_subscription_agent(*, pass_auth_mode=True):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_credentials",
            return_value=SimpleNamespace(
                token="sk-ant-oat01-current-token",
                source="claude_code_credentials_file",
                auth_mode="subscription_only",
                ignored_sources=("ANTHROPIC_API_KEY",),
            ),
        ),
    ):
        kwargs = dict(
            api_key="sk-ant-oat01-current-token",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            api_mode="anthropic_messages",
            model="claude-haiku-4-5-20251001",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        if pass_auth_mode:
            kwargs["auth_mode"] = "subscription_only"
        return AIAgent(**kwargs)


def test_agent_preserves_subscription_mode_in_primary_runtime():
    agent = _make_subscription_agent()

    assert agent.auth_mode == "subscription_only"
    assert agent._anthropic_auth_mode == "subscription_only"
    assert agent._primary_runtime["auth_mode"] == "subscription_only"


def test_agent_infers_configured_subscription_mode_when_surface_drops_field():
    strict = SimpleNamespace(
        token="sk-ant-oat01-current-token",
        source="claude_code_credentials_file",
        auth_mode="subscription_only",
        ignored_sources=("ANTHROPIC_API_KEY",),
    )
    with (
        patch(
            "agent.anthropic_adapter.get_configured_anthropic_auth_mode",
            return_value="subscription_only",
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_credentials",
            return_value=strict,
        ) as strict_resolver,
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
    ):
        agent = AIAgent(
            api_key="sk-ant-api-must-be-ignored",
            base_url="https://api.anthropic.com",
            provider="anthropic",
            api_mode="anthropic_messages",
            model="claude-haiku-4-5-20251001",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    strict_resolver.assert_called_once_with(
        auth_mode="subscription_only",
        explicit_api_key="sk-ant-api-must-be-ignored",
    )
    assert agent.auth_mode == "subscription_only"
    assert agent._anthropic_api_key == "sk-ant-oat01-current-token"


def test_subscription_refresh_never_calls_default_token_resolver():
    agent = _make_subscription_agent()
    fresh = SimpleNamespace(
        token="sk-ant-oat01-fresh-token",
        source="claude_code_credentials_file",
        auth_mode="subscription_only",
        ignored_sources=("ANTHROPIC_API_KEY",),
    )

    with (
        patch(
            "agent.anthropic_adapter.resolve_anthropic_credentials",
            return_value=fresh,
        ) as strict_resolver,
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            side_effect=AssertionError("default resolver must not run"),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
    ):
        assert agent._try_refresh_anthropic_client_credentials() is True

    strict_resolver.assert_called_once_with(auth_mode="subscription_only")
    assert agent._anthropic_api_key == "sk-ant-oat01-fresh-token"
    assert agent._anthropic_auth_source == "claude_code_credentials_file"


def test_subscription_mode_skips_credential_pool_recovery():
    from agent.agent_runtime_helpers import recover_with_credential_pool

    agent = SimpleNamespace(
        provider="anthropic",
        auth_mode="subscription_only",
        _anthropic_auth_mode="subscription_only",
        _credential_pool=MagicMock(provider="anthropic"),
    )

    assert recover_with_credential_pool(
        agent,
        status_code=401,
        has_retried_429=False,
    ) == (False, False)
    agent._credential_pool.current.assert_not_called()


def test_auxiliary_anthropic_strict_mode_skips_pool_and_explicit_key():
    from agent import auxiliary_client as aux

    strict = SimpleNamespace(
        token="sk-ant-oat01-aux-token",
        source="claude_code_credentials_file",
        auth_mode="subscription_only",
        ignored_sources=("explicit_api_key",),
    )
    with (
        patch(
            "agent.anthropic_adapter.get_configured_anthropic_auth_mode",
            return_value="subscription_only",
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_credentials",
            return_value=strict,
        ) as strict_resolver,
        patch.object(
            aux,
            "_select_pool_entry",
            side_effect=AssertionError("strict mode must not inspect the pool"),
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
    ):
        client, _model = aux._try_anthropic(
            explicit_api_key="sk-ant-api-must-be-ignored"
        )

    strict_resolver.assert_called_once_with(
        auth_mode="subscription_only",
        explicit_api_key="sk-ant-api-must-be-ignored",
    )
    assert client.api_key == "sk-ant-oat01-aux-token"


def test_switch_model_keeps_strict_mode_and_never_loads_pool():
    agent = _make_subscription_agent()
    agent.auth_mode = "default"
    agent._anthropic_auth_mode = "default"
    agent._credential_pool = None
    strict = SimpleNamespace(
        token="sk-ant-oat01-switched-token",
        source="claude_code_credentials_file",
        auth_mode="subscription_only",
        ignored_sources=("explicit_api_key",),
    )

    with (
        patch(
            "agent.anthropic_adapter.get_configured_anthropic_auth_mode",
            return_value="subscription_only",
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_credentials",
            return_value=strict,
        ) as strict_resolver,
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
        patch("agent.credential_pool.load_pool") as load_pool,
    ):
        agent.switch_model(
            "claude-sonnet-4-6",
            "anthropic",
            api_key="sk-ant-api-must-be-ignored",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
        )

    load_pool.assert_not_called()
    strict_resolver.assert_called_once_with(
        auth_mode="subscription_only",
        explicit_api_key="sk-ant-api-must-be-ignored",
    )
    assert agent.auth_mode == "subscription_only"
    assert agent._anthropic_api_key == "sk-ant-oat01-switched-token"
    assert agent._primary_runtime["auth_mode"] == "subscription_only"


def test_primary_restore_restores_strict_metadata_without_pool_reselection():
    agent = _make_subscription_agent()
    agent._fallback_activated = True
    agent._rate_limited_until = 0
    agent.provider = "openrouter"
    agent.requested_provider = "openrouter"
    agent.auth_mode = "default"
    agent._anthropic_auth_mode = "default"
    agent._anthropic_auth_source = ""
    fallback_pool = MagicMock(provider="openrouter")
    agent._credential_pool = fallback_pool

    with (
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
        patch("agent.credential_pool.load_pool") as load_pool,
    ):
        assert agent._restore_primary_runtime() is True

    load_pool.assert_not_called()
    assert agent.provider == "anthropic"
    assert agent.auth_mode == "subscription_only"
    assert agent._anthropic_auth_mode == "subscription_only"
    assert agent._anthropic_auth_source == "claude_code_credentials_file"
    assert agent._credential_pool is None


def test_anthropic_fallback_keeps_subscription_mode_and_skips_pool():
    from agent.chat_completion_helpers import try_activate_fallback

    agent = MagicMock()
    agent.provider = "openrouter"
    agent.model = "openrouter/auto"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_mode = "chat_completions"
    agent.auth_mode = ""
    agent._anthropic_auth_mode = "default"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = [
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
    ]
    agent._primary_runtime = {
        "provider": "openrouter",
        "model": "openrouter/auto",
        "base_url": "https://openrouter.ai/api/v1",
    }
    agent._credential_pool = MagicMock(provider="openrouter")
    agent._transport_cache = {}
    agent._config_context_length = None
    agent._is_azure_openai_url.return_value = False
    agent._is_direct_openai_url.return_value = False
    agent._provider_model_requires_responses_api.return_value = False
    agent._anthropic_prompt_cache_policy.return_value = (True, True)
    agent._ensure_lmstudio_runtime_loaded.return_value = None
    agent.context_compressor = None

    strict = SimpleNamespace(
        token="sk-ant-oat01-fallback-token",
        source="claude_code_credentials_file",
        auth_mode="subscription_only",
        ignored_sources=(),
    )
    fallback_client = SimpleNamespace(
        api_key="sk-ant-oat01-fallback-token",
        base_url="https://api.anthropic.com",
    )
    with (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fallback_client, "claude-haiku-4-5-20251001"),
        ),
        patch(
            "agent.anthropic_adapter.get_configured_anthropic_auth_mode",
            return_value="subscription_only",
        ),
        patch(
            "agent.anthropic_adapter.resolve_anthropic_credentials",
            return_value=strict,
        ) as strict_resolver,
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=MagicMock(),
        ),
        patch("agent.credential_pool.load_pool") as load_pool,
    ):
        assert try_activate_fallback(agent) is True

    load_pool.assert_not_called()
    strict_resolver.assert_called_once()
    assert agent.provider == "anthropic"
    assert agent.auth_mode == "subscription_only"
    assert agent._anthropic_api_key == "sk-ant-oat01-fallback-token"
    assert agent._credential_pool is None
