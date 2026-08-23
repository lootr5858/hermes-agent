"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, tokens_io, cost, cwd]

Available fields:
    model        — bare model id, vendor prefix dropped (``gpt-5.4``)
    context_pct  — last-call context occupancy as a percent (``5%``)
    latency      — wall-clock duration of the turn (``22s``, ``1m05s``)
    tokens_io    — message and conversation input/output tokens
    cost         — estimated API cost, ``included``, or ``unknown``
    cwd          — home-relative working dir (``~``)

``latency`` is opt-in: it is NOT in the default field set, so a footer whose
``fields`` are unset renders exactly as before.

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.

The footer is appended to the final response text in ``gateway/run.py`` right
before returning the response to the adapter send path — so it only lands on
the final message a user sees, not on tool-progress updates or streaming
partials.  When streaming is on and the final text has already been delivered
piecemeal, the footer is sent as a separate trailing message via
``send_trailing_footer()``.
"""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any, Iterable, Optional

from agent.usage_pricing import format_cost_label, format_token_count_compact

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def resolve_footer_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective runtime-footer config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False)
        2. ``display.runtime_footer``
        3. ``display.platforms.<platform_key>.runtime_footer``
    """
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]

    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    message_input_tokens: int = 0,
    message_output_tokens: int = 0,
    conversation_input_tokens: int = 0,
    conversation_output_tokens: int = 0,
    conversation_cost_usd: Optional[float] = None,
    conversation_cost_status: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "latency":
            # Wall-clock turn duration. Skipped when the caller supplied no
            # timing (call sites that don't measure) or the value is negative.
            if turn_seconds is not None and turn_seconds >= 0:
                parts.append(_format_latency(turn_seconds))
        elif field == "tokens_io":
            parts.append(
                "msg "
                f"↑{format_token_count_compact(message_input_tokens)} "
                f"↓{format_token_count_compact(message_output_tokens)} / total "
                f"↑{format_token_count_compact(conversation_input_tokens)} "
                f"↓{format_token_count_compact(conversation_output_tokens)}"
            )
        elif field == "cost":
            if conversation_cost_status == "included":
                parts.append("cost included")
            elif conversation_cost_status == "unknown":
                parts.append("cost unknown")
            elif conversation_cost_usd is not None:
                parts.append(
                    f"cost {format_cost_label(Decimal(str(conversation_cost_usd)))}"
                )
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    message_input_tokens: int = 0,
    message_output_tokens: int = 0,
    conversation_input_tokens: int = 0,
    conversation_output_tokens: int = 0,
    conversation_cost_usd: Optional[float] = None,
    conversation_cost_status: Optional[str] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.

    ``turn_seconds`` is the wall-clock duration of the agent run, measured by
    the caller with ``time.monotonic()``.  Callers that don't measure it leave
    it ``None`` and the ``latency`` field is skipped.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        turn_seconds=turn_seconds,
        message_input_tokens=message_input_tokens,
        message_output_tokens=message_output_tokens,
        conversation_input_tokens=conversation_input_tokens,
        conversation_output_tokens=conversation_output_tokens,
        conversation_cost_usd=conversation_cost_usd,
        conversation_cost_status=conversation_cost_status,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
    )


def conversation_usage(session_db: Any, session_id: str) -> dict[str, Any]:
    """Return usage for one conversation across compression continuations."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "cost_status": None,
    }
    statuses: set[str] = set()
    for lineage_id in session_db.get_compression_lineage(session_id):
        row = session_db.get_session(lineage_id)
        if not row:
            continue
        input_tokens = int(row.get("input_tokens") or 0)
        output_tokens = int(row.get("output_tokens") or 0)
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += output_tokens
        raw_cost = row.get("actual_cost_usd")
        if raw_cost is None:
            raw_cost = row.get("estimated_cost_usd")
        totals["cost_usd"] += float(raw_cost or 0)
        if input_tokens or output_tokens or int(row.get("api_call_count") or 0):
            statuses.add(str(row.get("cost_status") or "unknown"))

    if "unknown" in statuses:
        totals["cost_status"] = "unknown"
    elif statuses - {"included"}:
        totals["cost_status"] = "estimated"
    elif "included" in statuses:
        totals["cost_status"] = "included"
    return totals
