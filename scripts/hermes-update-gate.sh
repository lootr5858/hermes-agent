#!/usr/bin/env bash
# Deterministic weekly gate. Empty stdout means no upstream update.

set -uo pipefail

REPO="${HERMES_REPO:-$HOME/.hermes/hermes-agent}"
UPDATER="${HERMES_UPDATER:-$HOME/.hermes/bin/hermes-update-local}"
EXIT_REVIEW=10
DETECT=0
[[ "${1:-}" == "--detect" ]] && DETECT=1

fail_safe() {
    echo "⚠️ *Hermes update gate ERROR* — could not complete the weekly check."
    echo "Reason: $1"
    echo "No changes applied. Re-run manually: \`cd $REPO && $UPDATER --evaluate\`"
    exit 0
}

strip_ansi() { sed -E 's/\x1b\[[0-9;]*m//g'; }

[[ -d "$REPO/.git" ]] || fail_safe "repo not found at $REPO"
[[ -x "$UPDATER" ]] || fail_safe "updater not found/executable at $UPDATER"
cd "$REPO" || fail_safe "cannot cd to $REPO"

PRE_SHA="$(git rev-parse local/working 2>/dev/null)" || fail_safe "cannot read local/working SHA"
git fetch -q upstream 2>/dev/null || fail_safe "git fetch upstream failed"
DELTA="$(git rev-list --count main..upstream/main 2>/dev/null)" || fail_safe "cannot count delta"
[[ "$DELTA" -eq 0 ]] && exit 0

NEW_MAIN="$(git rev-parse --short upstream/main 2>/dev/null)"
OLD_MAIN="$(git rev-parse --short main 2>/dev/null)"
MAX_COMMITS=15
UPSTREAM_COMMITS="$(git log --oneline -n "$MAX_COMMITS" main..upstream/main 2>/dev/null)" || UPSTREAM_COMMITS=""
SHOWN_COMMITS="$(printf '%s\n' "$UPSTREAM_COMMITS" | grep -c .)"
OMITTED_COMMITS=$(( DELTA - SHOWN_COMMITS ))

EVAL_RAW="$("$UPDATER" --evaluate 2>&1)"
EVAL_STATUS=$?
[[ "$EVAL_STATUS" -eq 0 ]] || fail_safe "--evaluate failed (exit $EVAL_STATUS)"
EVAL_OUT="$(printf '%s\n' "$EVAL_RAW" | strip_ansi)"

DRY_RAW="$("$UPDATER" --dry-run 2>&1)"
DRY_STATUS=$?
if [[ "$DRY_STATUS" -ne 0 && "$DRY_STATUS" -ne "$EXIT_REVIEW" ]]; then
    fail_safe "--dry-run failed (exit $DRY_STATUS)"
fi
DRY_OUT="$(printf '%s\n' "$DRY_RAW" | strip_ansi)"

BUCKET_LINES="$(printf '%s\n' "$EVAL_OUT" | grep -E '^→ local/' || true)"
[[ -n "$BUCKET_LINES" ]] || fail_safe "no bucket lines parsed from --evaluate"

DROPPED_LINE="$(printf '%s\n' "$DRY_OUT" | grep -E 'dropped\s*:' | head -1 || true)"
DROPPED="$(printf '%s' "$DROPPED_LINE" | sed -E 's/.*dropped\s*:\s*//')"
HAS_DROP=0
[[ -n "$DROPPED" && "$DROPPED" != "(none)" ]] && HAS_DROP=1

FLAGGED="$(printf '%s\n' "$BUCKET_LINES" | grep -E '\[(COMPLEMENTARY_REVIEW|COMPLEMENTARY|GAP_HIDDEN|FULLY_COVERED|CONFLICT)\]' | grep -v 'COMPLEMENTARY_CLEAN' || true)"
WOULD_PUBLISH=0
printf '%s\n' "$DRY_OUT" | grep -qiE 'would publish|DRY RUN: would publish' && WOULD_PUBLISH=1

ROLLBACK="cd $REPO && git checkout local/working && git reset --hard $PRE_SHA"

if [[ "$DRY_STATUS" -eq "$EXIT_REVIEW" || -n "$FLAGGED" || "$HAS_DROP" -eq 1 || "$WOULD_PUBLISH" -eq 0 ]]; then
    echo "⚠️ *Hermes update — review needed* (+$DELTA commit(s), \`$OLD_MAIN\`→\`$NEW_MAIN\`)"
    echo
    if [[ -n "$UPSTREAM_COMMITS" ]]; then
        echo "*Upstream commits:*"
        printf '%s\n' "$UPSTREAM_COMMITS" | sed 's/^/• /'
        (( OMITTED_COMMITS > 0 )) && echo "… $OMITTED_COMMITS older commit(s) omitted"
        echo
    fi
    if [[ "$HAS_DROP" -eq 1 ]]; then
        echo "🔴 *Feature(s) would be DROPPED:* $DROPPED"
        echo
    fi
    if [[ "$WOULD_PUBLISH" -eq 0 && "$HAS_DROP" -eq 0 ]]; then
        echo "🔴 Dry-run did not reach a publishable candidate — manual look needed."
        echo
    fi
    if [[ -n "$FLAGGED" ]]; then
        echo "*Flagged features:*"
        printf '%s\n' "$EVAL_OUT" \
            | grep -A1 -E '\[(COMPLEMENTARY_REVIEW|COMPLEMENTARY|GAP_HIDDEN|FULLY_COVERED|CONFLICT)\]' \
            | grep -vE '^(--|.*COMPLEMENTARY_CLEAN)' \
            | sed -E 's/^→ /• /; s/^    /    ↳ /' || true
        echo
    fi
    echo "*Next step:* forward-port the specific conflicting or stale feature shown above, then rerun the gate."
    echo
    echo "_Not applied. Live checkout untouched._"
    exit 0
fi

GREEN_SUMMARY="$(printf '%s\n' "$BUCKET_LINES" | sed -E 's/^→ /• /')"
NFEAT="$(printf '%s\n' "$BUCKET_LINES" | grep -c .)"

if [[ "$DETECT" -eq 1 ]]; then
    echo "✅ *[DETECT] Would auto-update* (+$DELTA commit(s), \`$OLD_MAIN\`→\`$NEW_MAIN\`)"
    echo
    echo "All $NFEAT feature(s) green:"
    printf '%s\n' "$GREEN_SUMMARY"
    echo "(detect mode — nothing applied)"
    exit 0
fi

APPLY_OUT="$("$UPDATER" 2>&1)" || fail_safe "auto-apply failed — review manually. Pre-apply SHA: $PRE_SHA"
POST_SHA="$(git rev-parse --short local/working 2>/dev/null)"
DEPS_FLAG="no"
printf '%s\n' "$APPLY_OUT" | grep -qiE 'uv pip install|dependenc|requirements' && DEPS_FLAG="yes"

echo "✅ *Hermes auto-updated* (+$DELTA commit(s), main \`$OLD_MAIN\`→\`$NEW_MAIN\`)"
echo "All $NFEAT feature(s) kept. New working tip: \`$POST_SHA\`."
echo "Deps changed: $DEPS_FLAG"
echo
echo "*Rollback:*"
echo '```'
echo "$ROLLBACK"
[[ "$DEPS_FLAG" == "yes" ]] && echo "uv pip install -e .   # deps changed this run"
echo '```'
