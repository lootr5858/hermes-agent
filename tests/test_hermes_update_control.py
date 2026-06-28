from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
UPDATER = ROOT / "scripts" / "hermes-update-local"
GATE = ROOT / "scripts" / "hermes-update-gate.sh"


def _run(*args: str | Path, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(arg) for arg in args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = _run("git", *args, cwd=repo)
    assert result.returncode == 0, result.stderr
    return result


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if executable:
        path.chmod(0o755)


def _fixture_repo(tmp_path: Path, *, conflict: bool = True, failing_validation: bool = False) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Hermes Test")

    _write(repo / "run_agent.py", "VALUE = 'ok'\n")
    _write(repo / "agent" / "auxiliary_client.py", "async def async_call_llm():\n    return 'base'\n")
    if failing_validation:
        _write(repo / "tests" / "agent" / "test_anthropic_subscription_only.py", "def test_placeholder():\n    assert True\n")
        _write(repo / "scripts" / "run_tests.sh", "#!/usr/bin/env bash\nexit 1\n", executable=True)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base")
    _git(repo, "branch", "local/working")

    _git(repo, "checkout", "-b", "local/predecessor")
    _write(repo / "predecessor.txt", "merged first\n")
    _git(repo, "add", "predecessor.txt")
    _git(repo, "commit", "-m", "predecessor")

    _git(repo, "checkout", "main")
    _git(repo, "checkout", "-b", "local/required")
    if conflict:
        _write(repo / "agent" / "auxiliary_client.py", "async def async_call_llm():\n    return 'feature'\n")
        _git(repo, "add", "agent/auxiliary_client.py")
    else:
        _write(repo / "required_feature.py", "ENABLED = True\n")
        _git(repo, "add", "required_feature.py")
    _git(repo, "commit", "-m", "required feature")
    _git(repo, "checkout", "main")

    origin = tmp_path / "origin.git"
    upstream = tmp_path / "upstream.git"
    assert _run("git", "clone", "--bare", repo, origin, cwd=tmp_path).returncode == 0
    assert _run("git", "clone", "--bare", repo, upstream, cwd=tmp_path).returncode == 0

    upstream_work = tmp_path / "upstream-work"
    assert _run("git", "clone", upstream, upstream_work, cwd=tmp_path).returncode == 0
    _git(upstream_work, "config", "user.email", "test@example.com")
    _git(upstream_work, "config", "user.name", "Hermes Test")
    if conflict:
        _write(upstream_work / "agent" / "auxiliary_client.py", "async def async_call_llm():\n    return 'upstream'\n")
        _git(upstream_work, "add", "agent/auxiliary_client.py")
    else:
        _write(upstream_work / "upstream_marker.py", "NEW = True\n")
        _git(upstream_work, "add", "upstream_marker.py")
    _git(upstream_work, "commit", "-m", "upstream change")
    _git(upstream_work, "push", "origin", "main")

    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "remote", "add", "upstream", str(upstream))
    _git(repo, "fetch", "upstream")

    python = repo / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (repo / ".git" / "info" / "exclude").write_text("venv/\n")

    features = tmp_path / "features.conf"
    features.write_text("local/predecessor\nlocal/required  !required\n")
    return repo, features


def _updater_env(repo: Path, features: Path, tmp_path: Path) -> dict[str, str]:
    return {
        **os.environ,
        "HERMES_REPO": str(repo),
        "HERMES_FEATURES_CONF": str(features),
        "HERMES_EVAL_DIR": str(tmp_path / "reports"),
    }


def _fake_review_updater(path: Path) -> Path:
    _write(
        path,
        """#!/usr/bin/env bash
case "$1" in
  --evaluate)
    printf '→ local/required   [CONFLICT]\\n    needs forward-port\\n'
    exit 0
    ;;
  --dry-run)
    printf '   dropped : local/required (conflict)\\n'
    printf 'DRY RUN: would ABORT publish and keep current local/working unchanged.\\n'
    exit 10
    ;;
  *)
    printf 'unexpected apply\\n'
    exit 99
    ;;
esac
""",
        executable=True,
    )
    return path


def test_required_feature_drop_returns_review_status(tmp_path: Path) -> None:
    repo, features = _fixture_repo(tmp_path)
    result = _run(UPDATER, "--dry-run", cwd=repo, env=_updater_env(repo, features, tmp_path))

    assert result.returncode == 10, result.stdout + result.stderr
    assert "REQUIRED feature(s) did not survive" in result.stdout + result.stderr


def test_gate_treats_review_status_as_review_not_error(tmp_path: Path) -> None:
    repo, _ = _fixture_repo(tmp_path)
    fake_updater = _fake_review_updater(tmp_path / "fake-updater")
    env = {**os.environ, "HERMES_REPO": str(repo), "HERMES_UPDATER": str(fake_updater)}

    result = _run(GATE, "--detect", cwd=repo, env=env)

    assert result.returncode == 0, result.stderr
    assert "Hermes update — review needed" in result.stdout
    assert "Feature(s) would be DROPPED" in result.stdout
    assert "gate ERROR" not in result.stdout


def test_gate_caps_large_upstream_changelog(tmp_path: Path) -> None:
    repo, _ = _fixture_repo(tmp_path)
    upstream_url = _git(repo, "remote", "get-url", "upstream").stdout.strip()
    upstream_work = tmp_path / "many-upstream-commits"
    assert _run("git", "clone", upstream_url, upstream_work, cwd=tmp_path).returncode == 0
    _git(upstream_work, "config", "user.email", "test@example.com")
    _git(upstream_work, "config", "user.name", "Hermes Test")
    for number in range(30):
        _git(upstream_work, "commit", "--allow-empty", "-m", f"upstream-{number}")
    _git(upstream_work, "push", "origin", "main")

    fake_updater = _fake_review_updater(tmp_path / "fake-updater")
    env = {**os.environ, "HERMES_REPO": str(repo), "HERMES_UPDATER": str(fake_updater)}
    result = _run(GATE, "--detect", cwd=repo, env=env)

    assert result.stdout.count("• ") <= 20
    assert "older commit(s) omitted" in result.stdout


def test_evaluator_prioritizes_real_conflict_over_hidden_gap(tmp_path: Path) -> None:
    repo, features = _fixture_repo(tmp_path)
    probes = tmp_path / "probes.conf"
    probes.write_text(
        "local/predecessor | predecessor | |\n"
        "local/required | anthropic | | async_call_llm\n"
    )
    env = _updater_env(repo, features, tmp_path)
    env["HERMES_PROBES_CONF"] = str(probes)

    result = _run(UPDATER, "--evaluate", cwd=repo, env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if "→ local/required" in line)
    plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
    assert "[CONFLICT]" in plain


def test_evaluator_excludes_upstream_commits_already_in_forward_port(tmp_path: Path) -> None:
    repo, features = _fixture_repo(tmp_path, conflict=False)
    _git(repo, "checkout", "-b", "local/forward-port", "upstream/main")
    _write(repo / "feature_only.py", "CUSTOM = True\n")
    _git(repo, "add", "feature_only.py")
    _git(repo, "commit", "-m", "forward port custom feature")
    _git(repo, "checkout", "main")
    features.write_text("local/forward-port\n")
    probes = tmp_path / "probes.conf"
    probes.write_text("local/forward-port | upstream change | |\n")
    env = _updater_env(repo, features, tmp_path)
    env["HERMES_PROBES_CONF"] = str(probes)

    result = _run(UPDATER, "--evaluate", cwd=repo, env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    summary = next(line for line in result.stdout.splitlines() if "→ local/forward-port" in line)
    assert "[INDEPENDENT]" in re.sub(r"\x1b\[[0-9;]*m", "", summary)
    details = next(line for line in result.stdout.splitlines() if "files " in line)
    assert "files 1 | overlap 0" in details


def test_probe_manifest_does_not_claim_logging_or_prompt_text_is_replaced() -> None:
    rows = {
        line.split("|", 1)[0].strip(): [part.strip() for part in line.split("|")]
        for line in (ROOT / "scripts" / "hermes-feature-probes.conf").read_text().splitlines()
        if line.startswith("local/")
    }

    assert rows["local/aux-async-logging-v2"][3] == ""
    assert rows["local/proactivity-boundaries-v2"][2] == ""
    assert rows["local/update-tooling"][1] == ""


def test_teach_applies_configured_predecessors_before_target(tmp_path: Path) -> None:
    repo, features = _fixture_repo(tmp_path)
    result = _run(UPDATER, "--teach", "local/required", cwd=repo, env=_updater_env(repo, features, tmp_path))
    combined = result.stdout + result.stderr
    match = re.search(r"^  1\) cd (.+)$", combined, re.MULTILINE)
    assert match, combined
    teach_worktree = Path(match.group(1))
    try:
        assert (teach_worktree / "predecessor.txt").read_text() == "merged first\n"
    finally:
        _run("git", "worktree", "remove", "--force", teach_worktree, cwd=repo)
        _run("git", "branch", "-D", "_teach_local-required", cwd=repo)


def test_targeted_validation_failure_blocks_required_candidate(tmp_path: Path) -> None:
    repo, features = _fixture_repo(tmp_path, conflict=False, failing_validation=True)
    result = _run(UPDATER, "--dry-run", cwd=repo, env=_updater_env(repo, features, tmp_path))

    assert result.returncode == 10, result.stdout + result.stderr
    assert "targeted validation failed" in result.stdout + result.stderr
