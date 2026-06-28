#!/usr/bin/env python3
"""hermes-eval-judge — Phase 2 LLM adjudicator for `hermes-update-local --judge`.

Reads an evidence directory (written by hermes-update-local) and asks the fork's
ALREADY-CONFIGURED auxiliary LLM provider to classify ONE custom feature against
the incoming upstream delta, then prints a short verdict to stdout.

Design / safety:
  * Pure completion via agent.auxiliary_client.async_call_llm — NO agent loop,
    NO tools, NO file/network access beyond the single LLM call, and a CLEAN
    minimal context (no AGENTS.md / memory / skills).
  * Uses the configured provider via auto-detection (task=None) — no new endpoint.
  * FAIL-CLOSED: any failure (no provider, timeout, non-JSON, invalid bucket)
    exits non-zero with nothing on stdout, so the caller leaves the feature
    NEEDS_HUMAN. Never mutates anything; advisory text only.

Input: argv[1] = a directory of plain-text files (avoids all shell-quoting):
  feature.txt bucket.txt owned.txt reroutes.txt reroute_changed.txt
  coverage.txt overlap.txt changelog.txt feature.diff [upstream.diff]
Output (stdout, success): a few human-readable lines; also writes verdict.json.
Env: HERMES_JUDGE_TIMEOUT (seconds, default 120), HERMES_JUDGE_MAXDIFF (chars).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys

VALID = {"FULLY_COVERED", "COMPLEMENTARY", "GAP_HIDDEN", "CONFLICT"}
MAX_DIFF = int(os.getenv("HERMES_JUDGE_MAXDIFF", "16000"))
TIMEOUT = float(os.getenv("HERMES_JUDGE_TIMEOUT", "120"))

INSTRUCTIONS = (
    "You are a meticulous software-merge analyst. A user maintains custom feature branches on "
    "top of an upstream project (Hermes Agent). For ONE feature, decide how it relates to the "
    "INCOMING upstream changes and pick exactly one bucket:\n"
    "- FULLY_COVERED: upstream now implements this feature's behaviour natively; the custom code "
    "is redundant and is a RETIRE candidate.\n"
    "- COMPLEMENTARY: feature and upstream changed the same area for DIFFERENT purposes; keep "
    "both (possibly re-anchor the feature's hunks).\n"
    "- GAP_HIDDEN: upstream improved code the feature overrides/reroutes, so the custom override "
    "would SHADOW the upstream improvement even after a clean merge — slim the feature (drop what "
    "upstream now does) or forward-port the upstream change.\n"
    "- CONFLICT: the only real issue is a mechanical merge conflict; no semantic coverage/shadowing.\n\n"
    "Weigh the evidence yourself. A deterministic prefilter's guess is given; you MAY overrule it "
    "with justification. Respond with ONLY one JSON object — no prose, no markdown, no tool use:\n"
    '{"bucket":"<one of the four>","confidence":<0.0-1.0>,'
    '"rationale":"<=60 words","recommendation":"<=40 words, a concrete action"}'
)


def _read(d: str, name: str) -> str:
    try:
        with open(os.path.join(d, name), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except Exception:
        return ""


def main() -> int:
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        sys.stderr.write("judge: usage: hermes-eval-judge <evidence-dir>\n")
        return 2
    d = sys.argv[1]

    feature = _read(d, "feature.txt").strip()
    det_bucket = _read(d, "bucket.txt").strip().upper()
    bundle = (
        f"FEATURE: {feature}\n"
        f"Prefilter bucket: {det_bucket}\n"
        f"Feature's distinctive owned symbols: {_read(d, 'owned.txt').strip() or '(none)'}\n"
        f"Upstream symbols the feature reroutes/overrides: {_read(d, 'reroutes.txt').strip() or '(none)'}\n"
        f"Of those reroutes, upstream CHANGED: {_read(d, 'reroute_changed.txt').strip() or '(none)'}\n"
        f"Owned symbols already present in upstream: {_read(d, 'coverage.txt').strip() or '0/0'}\n"
        f"Files both sides touched: {_read(d, 'overlap.txt').strip() or '(none)'}\n"
        f"Matching upstream commit subjects:\n{_read(d, 'changelog.txt').strip() or '(none)'}\n\n"
        f"=== FEATURE DIFF (its own changes over the overlapping files) ===\n"
        f"{_read(d, 'feature.diff')[:MAX_DIFF] or '(none)'}\n\n"
        f"=== UPSTREAM DIFF (incoming changes over those same files) ===\n"
        f"{_read(d, 'upstream.diff')[:MAX_DIFF] or '(none)'}\n"
    )

    try:
        from agent.auxiliary_client import async_call_llm, extract_content_or_reasoning
    except Exception as exc:  # import path / env not set up
        sys.stderr.write(f"judge: auxiliary client unavailable: {exc}\n")
        return 3

    async def _call():
        return await async_call_llm(
            messages=[{"role": "user", "content": INSTRUCTIONS + "\n\n" + bundle}],
            temperature=0,
            max_tokens=700,
            timeout=TIMEOUT,
        )

    try:
        resp = asyncio.run(asyncio.wait_for(_call(), timeout=TIMEOUT + 15))
        text = (extract_content_or_reasoning(resp) or "").strip()
    except Exception as exc:  # no provider, timeout, network, etc.
        sys.stderr.write(f"judge: LLM call failed: {exc}\n")
        return 3
    if not text:
        sys.stderr.write("judge: empty LLM response\n")
        return 3

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        sys.stderr.write("judge: no JSON object in response\n")
        return 4
    try:
        verdict = json.loads(match.group(0))
    except Exception as exc:
        sys.stderr.write(f"judge: unparseable JSON: {exc}\n")
        return 4

    bucket = str(verdict.get("bucket", "")).strip().upper()
    if bucket not in VALID:
        sys.stderr.write(f"judge: invalid bucket {bucket!r}\n")
        return 4
    try:
        conf = max(0.0, min(1.0, float(verdict.get("confidence", 0))))
    except Exception:
        conf = 0.0
    rationale = " ".join(str(verdict.get("rationale", "")).split())
    rec = " ".join(str(verdict.get("recommendation", "")).split())

    try:
        with open(os.path.join(d, "verdict.json"), "w", encoding="utf-8") as fh:
            json.dump({"bucket": bucket, "confidence": round(conf, 2),
                       "rationale": rationale, "recommendation": rec}, fh)
    except Exception:
        pass

    agree = "agrees with prefilter" if bucket == det_bucket else f"OVERRULES prefilter ({det_bucket})"
    print(f"{bucket} (confidence {conf:.2f}; {agree})")
    if rationale:
        print(f"why: {rationale}")
    if rec:
        print(f"do:  {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
