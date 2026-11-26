"""The required CI job stays small: core dependencies (numpy) + pytest only.

Plain text checks with the standard library (no YAML parser in the core environment). The
workflow is written in block style, so a job is "the lines indented under its name" and a key is
"a line that starts with it".
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# 필수 잡이 설치·실행하면 안 되는 것 — 무거운 선택 의존성, 모델 다운로드, 실제 모델 테스트
FORBIDDEN_IN_CORE = ("--extra", "--all-extras", "chromadb", "fastembed", "EMBEDSTAMP_REAL")


def strip_comment(line: str) -> str:
    # 줄 맨 앞이나 공백 뒤의 # 부터는 YAML 주석이다(이 워크플로에는 따옴표 안 # 이 없다)
    return re.split(r"(?:^|\s)#", line, maxsplit=1)[0].rstrip()


def lines() -> list[str]:
    return [strip_comment(line) for line in WORKFLOW.read_text(encoding="utf-8").splitlines()]


def block(all_lines: list[str], header: str, indent: int) -> list[str]:
    """Lines under ``header`` (a line equal to ``' ' * indent + header``) until the next sibling key."""
    start = all_lines.index(" " * indent + header)
    body = []
    for line in all_lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip(" ")) <= indent:
            break
        body.append(line)
    return body


def triggers() -> set[str]:
    return {line.strip().rstrip(":") for line in block(lines(), "on:", 0) if re.match(r"^  \S", line)}


def job(name: str) -> list[str]:
    return block(block(lines(), "jobs:", 0), f"{name}:", 2)


def job_text(name: str) -> str:
    return "\n".join(job(name))


def run_commands(name: str) -> list[str]:
    """Every shell line of a job's ``run:`` steps (one-line and ``run: |`` blocks)."""
    body = job(name)
    commands = []
    for i, line in enumerate(body):
        m = re.match(r"^(\s*)(?:- )?run:\s*(.*)$", line)
        if not m:
            continue
        if m.group(2) not in ("|", ">"):
            commands.append(m.group(2))
            continue
        key_indent = len(m.group(1))
        for inner in body[i + 1 :]:
            if inner.strip() and len(inner) - len(inner.lstrip(" ")) <= key_indent:
                break
            if inner.strip():
                commands.append(inner.strip())
    return commands


def test_workflow_has_two_jobs():
    jobs = {line.strip().rstrip(":") for line in block(lines(), "jobs:", 0) if re.match(r"^  \S", line)}
    assert jobs == {"core", "extras"}


def test_triggers_are_push_pr_and_manual_without_schedule():
    assert triggers() == {"push", "pull_request", "workflow_dispatch"}
    assert "schedule" not in "\n".join(lines())


def test_core_job_runs_on_every_trigger():
    text = job_text("core")
    # if: 조건이나 needs: 가 붙으면 push·PR 에서 건너뛰거나 다른 잡에 묶일 수 있다
    assert not re.search(r"^    if:", text, re.M)
    assert not re.search(r"^    needs:", text, re.M)


def test_core_job_installs_and_runs_core_only():
    text = job_text("core")
    for word in FORBIDDEN_IN_CORE:
        assert word not in text, f"core job mentions {word!r}"
    commands = run_commands("core")
    assert "uv sync --locked" in commands
    assert 'uv run pytest -q -m "not timing"' in commands
    assert "uv run embedstamp demo" in commands
    assert "uv run python examples/quickstart.py" in commands


@pytest.mark.parametrize("name", ["core", "extras"])
def test_jobs_pin_uv_and_have_a_timeout(name):
    text = job_text(name)
    assert 'version: "0.13.6"' in text
    assert re.search(r"^    timeout-minutes: \d+$", text, re.M)


def test_extras_job_is_manual_and_requires_chromadb():
    text = job_text("extras")
    assert re.search(r"^    if: github\.event_name == 'workflow_dispatch'$", text, re.M)
    assert re.search(r'^      EMBEDSTAMP_REQUIRE_CHROMA: "1"$', text, re.M)
    commands = run_commands("extras")
    assert "uv sync --locked --extra chroma" in commands
    assert any(c.startswith("uv run --extra chroma pytest") and "-m chroma" in c for c in commands)


def test_markers_used_by_ci_are_registered():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for marker in ("chroma", "timing", "real"):
        assert f'"{marker}: ' in pyproject


def test_core_dev_group_has_no_optional_dependencies():
    import tomllib

    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dev = " ".join(data["dependency-groups"]["dev"])
    assert "chromadb" not in dev and "fastembed" not in dev
