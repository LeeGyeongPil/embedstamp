import subprocess
import sys
import time
from pathlib import Path

import pytest

from embedstamp.cli import main

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["overlap@10", "different", "self-recall@10", "switched", "same"]


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_demo_runs_offline(lang, capsys):
    # 내용·순서만 본다 — 시간 상한은 아래 timing 테스트가 따로 본다(느린 공유 러너에서 잘못 실패하지 않게)
    assert main(["--lang", lang, "demo"], env={}) == 0
    out = capsys.readouterr().out
    position = -1
    for token in ORDER:
        assert token in out, token
        # the first occurrence of each token must come after the previous one
        first = out.index(token)
        assert first > position, f"{token!r} appears out of order"
        position = first
    assert "practice mock" in out or "연습용 목" in out


@pytest.mark.timing
def test_demo_finishes_within_ten_seconds(capsys):
    # 첫 실행 체험이 지루하지 않은지 — 로컬 전체 테스트에서만 돈다(CI 필수 잡은 -m "not timing")
    started = time.monotonic()
    assert main(["--lang", "en", "demo"], env={}) == 0
    assert time.monotonic() - started < 10
    capsys.readouterr()


def test_demo_does_not_leave_files_behind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main(["demo"], env={})
    assert list(tmp_path.iterdir()) == []


def run_script(name):
    return subprocess.run(
        [sys.executable, str(ROOT / "examples" / name), *sys.argv[0:0]],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )


def test_quickstart_example_runs():
    res = run_script("quickstart.py")
    assert res.returncode == 0, res.stderr
    for token in ("same", "drift", "different", "EmbeddingMismatchError caught"):
        assert token in res.stdout


@pytest.mark.chroma
def test_chroma_example_runs_in_mock_mode(chromadb):
    res = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "chroma_guard.py"), "--mock"],
        capture_output=True, text=True, timeout=120, cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    assert "EmbeddingMismatchError caught" in res.stdout
