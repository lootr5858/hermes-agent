"""Tests for Telegram dispatch handoff buttons."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_dispatch_cli_raises_on_nonzero_exit():
    adapter = _make_adapter()
    process = SimpleNamespace(
        returncode=1,
        communicate=AsyncMock(return_value=(b"", b"dispatch failed")),
    )

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        with pytest.raises(RuntimeError, match="dispatch failed"):
            await adapter._dispatch_cli("list-pending")


@pytest.mark.asyncio
async def test_surface_pending_marks_notified_only_after_send():
    adapter = _make_adapter()
    events = []

    async def dispatch_cli(*args):
        events.append(args)
        if args == ("list-pending",):
            return '[{"id":"handoff-1","title":"Fix Hermes","target_dir":"/repo","source":"agent"}]'
        return "ok"

    async def send_message(**kwargs):
        events.append("sent")

    adapter._dispatch_cli = dispatch_cli
    adapter._bot.send_message = AsyncMock(side_effect=send_message)

    def button(text, callback_data):
        return SimpleNamespace(text=text, callback_data=callback_data)

    with (
        patch.dict("os.environ", {"DISPATCH_CHAT_ID": "12345"}),
        patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardButton",
            side_effect=button,
        ),
        patch(
            "plugins.platforms.telegram.adapter.InlineKeyboardMarkup",
            side_effect=lambda rows: rows,
        ),
    ):
        await adapter._surface_pending_handoffs()

    assert events == [("list-pending",), "sent", ("mark-notified", "handoff-1")]
    sent = adapter._bot.send_message.call_args.kwargs
    assert sent["chat_id"] == 12345
    assert "Fix Hermes" in sent["text"]
    callbacks = [button.callback_data for button in sent["reply_markup"][0]]
    assert callbacks == ["disp:a:handoff-1", "disp:s:handoff-1"]


@pytest.mark.asyncio
async def test_dispatch_callback_rejects_unauthorized_user():
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=False)
    adapter._run_dispatch_seed = AsyncMock()
    query = AsyncMock()
    query.data = "disp:a:handoff-1"
    query.message = SimpleNamespace(
        chat_id=12345,
        chat=SimpleNamespace(type="private"),
        message_thread_id=None,
    )
    query.from_user = SimpleNamespace(id=999, first_name="Mallory")
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, MagicMock())

    assert "not authorized" in query.answer.call_args.kwargs["text"].lower()
    adapter._run_dispatch_seed.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_skip_does_not_report_success_when_store_update_fails():
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    adapter._dispatch_cli = AsyncMock(side_effect=RuntimeError("store unavailable"))
    query = AsyncMock()
    query.data = "disp:s:handoff-1"
    query.message = SimpleNamespace(
        chat_id=12345,
        chat=SimpleNamespace(type="private"),
        message_thread_id=None,
    )
    query.from_user = SimpleNamespace(id=12345, first_name="Evan")
    update = SimpleNamespace(callback_query=query)

    await adapter._handle_callback_query(update, MagicMock())

    assert "failed" in query.answer.call_args.kwargs["text"].lower()
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_approve_still_seeds_when_message_edit_fails():
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=True)

    async def seed(*args):
        return None

    adapter._run_dispatch_seed = MagicMock(side_effect=seed)
    query = AsyncMock()
    query.data = "disp:a:handoff-1"
    query.message = SimpleNamespace(
        chat_id=12345,
        chat=SimpleNamespace(type="private"),
        message_thread_id=None,
    )
    query.from_user = SimpleNamespace(id=12345, first_name="Evan")
    query.edit_message_text = AsyncMock(side_effect=RuntimeError("message expired"))
    update = SimpleNamespace(callback_query=query)

    def close_task(coro):
        coro.close()
        return MagicMock()

    with patch("asyncio.create_task", side_effect=close_task) as create_task:
        await adapter._handle_callback_query(update, MagicMock())

    adapter._run_dispatch_seed.assert_called_once_with(12345, "handoff-1")
    create_task.assert_called_once()


def test_dispatch_poller_is_not_duplicated():
    adapter = _make_adapter()
    existing_task = MagicMock()
    existing_task.done.return_value = False
    adapter._dispatch_task = existing_task

    with patch("asyncio.create_task") as create_task:
        adapter._ensure_dispatch_poller()

    create_task.assert_not_called()


def test_dispatch_poller_can_be_disabled(monkeypatch):
    adapter = _make_adapter()
    monkeypatch.setenv("DISPATCH_ENABLED", "0")

    def close_task(coro):
        coro.close()
        return MagicMock()

    with patch("asyncio.create_task", side_effect=close_task) as create_task:
        adapter._ensure_dispatch_poller()

    create_task.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_cancels_and_awaits_dispatch_poller():
    adapter = _make_adapter()

    async def wait_forever():
        await asyncio.Future()

    task = asyncio.create_task(wait_forever())
    adapter._dispatch_task = task

    await adapter.disconnect()

    assert task.cancelled()
    assert adapter._dispatch_task is None
