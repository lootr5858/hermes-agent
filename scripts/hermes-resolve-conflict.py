#!/usr/bin/env python3
"""hermes-resolve-conflict — Phase 3 DRAFT conflict resolver for `--teach --draft`.

Given ONE conflicted file (with git merge markers) it proposes a resolution and
writes it to a separate DRAFT path. It NEVER edits the conflicted file itself —
the human reviews the draft, copies it over if good, and commits (which is what
records the rerere resolution, exactly as in plain --teach). Draft-only, advisory.

Safety / design:
  * Python owns ALL parsing + splicing. Non-conflicted lines are preserved
    BYTE-FOR-BYTE. For each hunk the LLM only chooses a MERGE STRATEGY
    (ours | theirs | union_ours_theirs | union_theirs_ours | custom); for every
    strategy except `custom`, Python emits the ORIGINAL conflict-side lines
    verbatim, so indentation can't drift. Only genuine interleaving uses model
    text (re-indented to the hunk's base + compile-checked).
  * Strategy chosen by a PURE completion via agent.auxiliary_client.async_call_llm
    — no agent loop, no tools, configured provider via auto-detect, no new endpoint.
  * FAIL-CLOSED: any failure (no provider, timeout, unparseable, residual markers)
    exits non-zero and writes NO draft; the caller falls back to manual resolution.
  * Validates the draft: zero residual markers; for *.py, py_compile is reported.

Usage:  hermes-resolve-conflict <conflicted-file> <draft-out> <ours-label> <theirs-label>
Env: HERMES_JUDGE_TIMEOUT (default 120s), HERMES_RESOLVE_CTX (context lines, default 14).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

TIMEOUT = float(os.getenv("HERMES_JUDGE_TIMEOUT", "120"))
CTX = int(os.getenv("HERMES_RESOLVE_CTX", "14"))

RE_START = re.compile(r"^<{7}")
RE_BASE = re.compile(r"^\|{7}")
RE_MID = re.compile(r"^={7}\s*$")
RE_END = re.compile(r"^>{7}")
RE_ANY_MARKER = re.compile(r"^(<{7}|\|{7}|={7}\s*$|>{7})", re.MULTILINE)

INSTRUCTIONS = (
    "Resolve ONE git merge conflict. OURS (HEAD) is the UPSTREAM version; THEIRS is the user's "
    "CUSTOM feature ({label}). Choose how to merge, strongly preferring to PRESERVE BOTH intents. "
    "Respond with ONLY one JSON object — no prose, no markdown fences:\n"
    '{{"strategy":"<ours|theirs|union_ours_theirs|union_theirs_ours|custom>",'
    '"custom_replacement":"<ONLY when strategy==custom: the exact replacement lines; else empty>"}}\n'
    "- ours / theirs: keep only that side (one side fully supersedes the other).\n"
    "- union_ours_theirs / union_theirs_ours: keep BOTH sides verbatim in that order — the COMMON "
    "case when each side adds independent code at the same spot. Prefer this over custom.\n"
    "- custom: ONLY when the two must be interwoven into genuinely new code; then put the merged "
    "lines (correct indentation, no markers) in custom_replacement."
)


def _parse(lines):
    """Split into ('text', raw_lines) and ('conflict', ours_lines, theirs_lines) segments."""
    segs, buf, i, n = [], [], 0, len(lines)
    while i < n:
        if RE_START.match(lines[i]):
            if buf:
                segs.append(("text", buf)); buf = []
            ours, theirs = [], []
            i += 1
            while i < n and not RE_BASE.match(lines[i]) and not RE_MID.match(lines[i]):
                ours.append(lines[i]); i += 1
            if i < n and RE_BASE.match(lines[i]):          # diff3 base — skip
                i += 1
                while i < n and not RE_MID.match(lines[i]):
                    i += 1
            if i < n and RE_MID.match(lines[i]):
                i += 1
            while i < n and not RE_END.match(lines[i]):
                theirs.append(lines[i]); i += 1
            if i < n and RE_END.match(lines[i]):
                i += 1
            segs.append(("conflict", ours, theirs))
        else:
            buf.append(lines[i]); i += 1
    if buf:
        segs.append(("text", buf))
    return segs


def _strip_fences(text):
    t = text.strip("\n")
    m = re.match(r"^```[^\n]*\n(.*)\n```$", t, re.DOTALL)
    return m.group(1) if m else t


def _base_indent(*line_lists):
    indents = []
    for ll in line_lists:
        for raw in ll:
            s = raw.rstrip("\n")
            if s.strip():
                indents.append(len(s) - len(s.lstrip(" ")))
    return min(indents) if indents else 0


def _reindent(text, base):
    lines = text.split("\n")
    nonblank = [l for l in lines if l.strip()]
    if not nonblank:
        return text
    cur = min(len(l) - len(l.lstrip(" ")) for l in nonblank)
    delta = base - cur
    out = []
    for l in lines:
        if not l.strip():
            out.append(l)
        elif delta >= 0:
            out.append(" " * delta + l)
        else:
            drop = min(-delta, len(l) - len(l.lstrip(" ")))
            out.append(l[drop:])
    return "\n".join(out)


async def _decide(ours, theirs, before, after, label):
    from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
    prompt = (
        INSTRUCTIONS.format(label=label) +
        "\n\n--- context BEFORE ---\n" + before +
        "\n--- OURS (upstream) ---\n" + ours +
        "\n--- THEIRS (feature) ---\n" + theirs +
        "\n--- context AFTER ---\n" + after + "\n\nJSON:"
    )
    resp = await asyncio.wait_for(
        async_call_llm(messages=[{"role": "user", "content": prompt}],
                       temperature=0, max_tokens=1500, timeout=TIMEOUT),
        timeout=TIMEOUT + 15,
    )
    text = (extract_content_or_reasoning(resp) or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON in strategy response")
    return json.loads(m.group(0))


def main() -> int:
    if len(sys.argv) < 5:
        sys.stderr.write("resolve: usage: <conflicted-file> <draft-out> <ours-label> <theirs-label>\n")
        return 2
    src, out, _ours_label, theirs_label = sys.argv[1:5]
    try:
        with open(src, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except Exception as exc:
        sys.stderr.write(f"resolve: cannot read {src}: {exc}\n")
        return 2

    segs = _parse(raw.splitlines(keepends=True))
    if not any(s[0] == "conflict" for s in segs):
        sys.stderr.write("resolve: no conflict markers found\n")
        return 2
    try:
        import agent.auxiliary_client  # noqa: F401 — fail early if env unset
    except Exception as exc:
        sys.stderr.write(f"resolve: auxiliary client unavailable: {exc}\n")
        return 3

    out_lines, hunk, strategies = [], 0, []
    for idx, seg in enumerate(segs):
        if seg[0] == "text":
            out_lines.extend(seg[1]); continue
        _, ours, theirs = seg
        before = "".join(segs[idx - 1][1][-CTX:]) if idx > 0 and segs[idx - 1][0] == "text" else ""
        after = "".join(segs[idx + 1][1][:CTX]) if idx + 1 < len(segs) and segs[idx + 1][0] == "text" else ""
        try:
            verdict = asyncio.run(_decide("".join(ours), "".join(theirs), before, after, theirs_label))
        except Exception as exc:
            sys.stderr.write(f"resolve: LLM call failed on hunk {hunk + 1}: {exc}\n")
            return 3
        strat = str(verdict.get("strategy", "")).strip().lower()
        if strat == "ours":
            repl = list(ours)
        elif strat == "theirs":
            repl = list(theirs)
        elif strat == "union_ours_theirs":
            repl = list(ours) + list(theirs)
        elif strat == "union_theirs_ours":
            repl = list(theirs) + list(ours)
        elif strat == "custom":
            custom = _strip_fences(str(verdict.get("custom_replacement", "")))
            if not custom.strip() or RE_ANY_MARKER.search(custom):
                sys.stderr.write(f"resolve: empty/invalid custom replacement on hunk {hunk + 1}\n")
                return 4
            body = _reindent(custom, _base_indent(ours, theirs))
            if not body.endswith("\n"):
                body += "\n"
            repl = body.splitlines(keepends=True)
        else:
            sys.stderr.write(f"resolve: unknown strategy {strat!r} on hunk {hunk + 1}\n")
            return 4
        out_lines.extend(repl)
        strategies.append(strat)
        hunk += 1

    draft = "".join(out_lines)
    if RE_ANY_MARKER.search(draft):
        sys.stderr.write("resolve: residual conflict markers in draft\n")
        return 4
    try:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(draft)
    except Exception as exc:
        sys.stderr.write(f"resolve: cannot write draft {out}: {exc}\n")
        return 2

    note = ""
    if src.endswith(".py"):
        import py_compile
        try:
            py_compile.compile(out, doraise=True)
            note = "; py_compile OK"
        except Exception as exc:
            note = f"; py_compile FAILED ({type(exc).__name__}) — review carefully"
    print(f"resolved {hunk} hunk(s) [{', '.join(strategies)}]{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
