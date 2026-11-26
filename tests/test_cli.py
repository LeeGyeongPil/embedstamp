import json
import subprocess
import sys

import pytest

from embedstamp import FakeEmbedder, LocalIndex, stamp
from embedstamp.cli import main
from embedstamp.demo import load_lines

ENV = {"EMBEDSTAMP_LANG": "en"}


@pytest.fixture
def index_dir(tmp_path):
    docs = load_lines("demo_corpus.txt")[:30]
    embed = FakeEmbedder("fake:a")
    idx = LocalIndex(tmp_path / "idx")
    idx.add([f"d{i}" for i in range(len(docs))], docs, embed(docs))
    idx.save(stamp(embed, model="fake:a"))
    return tmp_path / "idx"


def run(*argv):
    return main(list(argv), env=ENV)


@pytest.mark.parametrize(
    ("embedder", "code"),
    [("fake:a", 0), ("fake:a?noise=0.05", 10), ("fake:b", 11), ("fake:a?dim=256", 11)],
)
def test_check_exit_codes(index_dir, embedder, code, capsys):
    assert run("check", "--store", f"local://{index_dir}", "--embedder", embedder) == code
    assert "status:" in capsys.readouterr().out


def test_check_without_fingerprint_exits_12(tmp_path, capsys):
    (tmp_path / "bare").mkdir()
    assert run("check", "--store", f"local://{tmp_path / 'bare'}", "--embedder", "fake:a") == 12
    assert "embedstamp stamp" in capsys.readouterr().err


def test_check_json_output(index_dir, capsys):
    code = run("check", "--store", f"local://{index_dir}", "--embedder", "fake:b", "--json")
    data = json.loads(capsys.readouterr().out)
    assert code == 11
    assert data["status"] == "different"
    assert data["min_cos"] < 0.9
    assert "probe_cosine_low" in data["reasons"]


def test_show_prints_model_dimension_and_time(index_dir, capsys):
    assert run("show", "--store", f"local://{index_dir}") == 0
    out = capsys.readouterr().out
    assert "model: fake:a" in out and "dimensions: 384" in out and "created: 20" in out
    assert run("show", "--store", f"local://{index_dir}", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["model"] == "fake:a" and data["dim"] == 384 and data["created_at"]


def test_stamp_then_check_via_sidecar(tmp_path, capsys):
    store = f"sidecar://{tmp_path / 'x.embedstamp.json'}"
    assert run("stamp", "--store", store, "--embedder", "fake:a", "--compact") == 0
    assert "Stamped" in capsys.readouterr().out
    assert run("check", "--store", store, "--embedder", "fake:a") == 0
    assert run("check", "--store", store, "--embedder", "fake:b") == 11


def test_stamp_save_failure_exits_4(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert run("stamp", "--store", f"sidecar://{blocker}/sub/fp.json", "--embedder", "fake:a") == 4
    assert "error" in capsys.readouterr().err


def test_migrate_with_switch_and_refusal(index_dir, tmp_path, capsys):
    store = f"local://{index_dir}"
    queries = tmp_path / "q.txt"
    queries.write_text("bread dough\nhiking boots\n", encoding="utf-8")
    code = run("migrate", "--store", store, "--to", "fake:b", "--from", "fake:a",
               "--queries", str(queries), "--switch", "--min-self-recall", "1.01")
    assert code == 13
    assert "Refused" in capsys.readouterr().err
    code = run("migrate", "--store", store, "--to", "fake:b", "--from", "fake:a",
               "--queries", str(queries), "--switch")
    out = capsys.readouterr().out
    assert code == 0
    assert "self-recall@10=" in out and "overlap@10" in out and "switched:" in out and "status: same" in out
    assert run("check", "--store", store, "--embedder", "fake:b") == 0


@pytest.mark.parametrize(("lang", "hint"), [("en", "LocalIndex.add"), ("ko", "LocalIndex.add")])
def test_migrate_on_stamp_only_index_explains_what_to_do(tmp_path, capsys, lang, hint):
    # F-3: README 4절처럼 stamp 만 한 local:// 인덱스에는 문서가 없다
    store = f"local://{tmp_path / 'idx'}"
    env = {"EMBEDSTAMP_LANG": lang}
    assert main(["stamp", "--store", store, "--embedder", "fake:a"], env=env) == 0
    capsys.readouterr()
    assert main(["migrate", "--store", store, "--to", "fake:b"], env=env) == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # "진행 중" 줄을 찍기 전에 멈춘다
    assert hint in captured.err and "examples/migrate_local.py" in captured.err
    assert ("embedstamp stamp" in captured.err) and (("error:" in captured.err) or ("오류:" in captured.err))


def test_errors_come_after_earlier_stdout_lines_when_piped(index_dir):
    # F-3: stdout 을 파이프로 받을 때도 오류(stderr)가 앞서 찍은 진행 줄 뒤에 나온다
    code = (
        "import sys; from embedstamp.cli import main; "
        f"sys.exit(main(['migrate', '--store', 'local://{index_dir}', '--to', 'fake:b', "
        "'--switch', '--min-self-recall', '1.01'], env={'EMBEDSTAMP_LANG': 'en'}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    assert proc.returncode == 13
    out = proc.stdout
    assert out.index("Re-embedding") < out.index("30/30 documents") < out.index("Refused to switch")


def test_rerun_switch_after_green_is_filled_shows_real_numbers(index_dir, capsys):
    # F-4: green 을 먼저 채우고(--switch 없이) 다시 --switch 로 돌리면 재사용 사실과 실제 수치를 보여 준다
    store = f"local://{index_dir}"
    assert run("migrate", "--store", store, "--to", "fake:b", "--from", "fake:a") == 0
    assert "green=idx__green" in capsys.readouterr().out
    assert run("migrate", "--store", store, "--to", "fake:b", "--switch") == 0
    out = capsys.readouterr().out
    assert "embedded=0 skipped=30 green=idx__green" in out
    assert "already in idx__green" in out
    assert "min_cos=-" not in out and "mean_cos=-" not in out
    assert "status: same  min_cos=1.0000  mean_cos=1.0000" in out
    assert "switched: idx now serves fake:b" in out


def test_usage_errors(capsys):
    assert run("check", "--store", "bogus", "--embedder", "fake:a") == 2
    assert run("check", "--store", "local://x", "--embedder", "nope") == 2
    assert run("migrate", "--store", "sidecar://x.json", "--to", "fake:b") == 2
    assert run("--bogus") == 2
    assert "error" in capsys.readouterr().err


@pytest.mark.chroma
def test_check_on_missing_chroma_dir_creates_nothing(tmp_path, capsys, chromadb):
    target = tmp_path / "nope"
    code = main(["check", "--store", f"chroma://{target}/x", "--embedder", "fake:a"], env={})
    assert code != 0 and not target.exists()
    assert "not found" in capsys.readouterr().err


def test_show_on_non_utf8_fingerprint_gives_fix_hint(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_bytes(b"\xff\xfe\x00")
    code = main(["show", "--store", f"sidecar://{bad}"], env={})
    assert code == 2
    assert "re-create it with embedstamp.stamp()" in capsys.readouterr().err
