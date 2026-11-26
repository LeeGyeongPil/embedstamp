import json
import re
import string
from importlib import resources

import pytest

from embedstamp import FakeEmbedder, LocalIndex, stamp
from embedstamp.cli import main
from embedstamp.i18n import Translator, resolve_lang

HANGUL = re.compile(r"[가-힣]")


def load(lang):
    text = resources.files("embedstamp").joinpath("locales", f"{lang}.json").read_text(encoding="utf-8")
    return json.loads(text)


def test_locale_files_have_identical_keys_and_placeholders():
    en, ko = load("en"), load("ko")
    assert set(en) == set(ko)
    fields = lambda s: {f for _, f, _, _ in string.Formatter().parse(s) if f}  # noqa: E731
    for key in en:
        assert fields(en[key]) == fields(ko[key]), key


@pytest.mark.parametrize(
    ("cli", "env", "expected"),
    [
        ("ko", {"EMBEDSTAMP_LANG": "en", "LC_ALL": "en_US.UTF-8"}, "ko"),
        ("en", {"EMBEDSTAMP_LANG": "ko"}, "en"),
        (None, {"EMBEDSTAMP_LANG": "ko", "LC_ALL": "en_US.UTF-8"}, "ko"),
        (None, {"EMBEDSTAMP_LANG": "en", "LC_ALL": "ko_KR.UTF-8"}, "en"),
        (None, {"LC_ALL": "ko_KR.UTF-8", "LC_MESSAGES": "en_US", "LANG": "en_US"}, "ko"),
        (None, {"LC_ALL": "", "LC_MESSAGES": "ko_KR", "LANG": "en_US"}, "ko"),
        (None, {"LC_MESSAGES": "en_US", "LANG": "ko_KR.UTF-8"}, "en"),
        (None, {"LANG": "ko_KR.UTF-8"}, "ko"),
        (None, {"LANG": "fr_FR.UTF-8"}, "en"),
        (None, {"LANG": "C"}, "en"),
        (None, {}, "en"),
        ("fr", {}, "en"),
    ],
)
def test_language_priority(cli, env, expected):
    assert resolve_lang(cli, env) == expected


def test_translator_switches_language():
    assert Translator("en")("cmd.demo") != Translator("ko")("cmd.demo")
    with pytest.raises(ValueError, match="unsupported language"):
        Translator("fr")


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_help_is_in_the_chosen_language(lang, capsys):
    assert main(["--lang", lang, "--help"], env={"LANG": "ko_KR.UTF-8" if lang == "en" else "C"}) == 0
    out = capsys.readouterr().out
    assert bool(HANGUL.search(out)) == (lang == "ko")
    assert main(["check", "--lang", lang, "--help"], env={}) == 0
    assert bool(HANGUL.search(capsys.readouterr().out)) == (lang == "ko")


@pytest.mark.parametrize("lang", ["en", "ko"])
def test_check_and_demo_output_follow_the_language(lang, tmp_path, capsys):
    embed = FakeEmbedder("fake:a")
    idx = LocalIndex(tmp_path / "idx")
    idx.add(["a", "b"], ["x", "y"], embed(["x", "y"]))
    idx.save(stamp(embed, model="fake:a"))
    main(["--lang", lang, "check", "--store", f"local://{tmp_path / 'idx'}", "--embedder", "fake:b"], env={})
    check_out = capsys.readouterr().out
    main(["--lang", lang, "demo"], env={})
    demo_out = capsys.readouterr().out
    for text in (check_out, demo_out):
        assert bool(HANGUL.search(text)) == (lang == "ko")
        assert "status:" in text  # status codes stay English in both languages


def test_environment_variable_selects_language(capsys):
    main(["--help"], env={"EMBEDSTAMP_LANG": "ko"})
    assert HANGUL.search(capsys.readouterr().out)
    main(["--help"], env={"EMBEDSTAMP_LANG": "en", "LANG": "ko_KR.UTF-8"})
    assert not HANGUL.search(capsys.readouterr().out)


@pytest.mark.parametrize("argv", [["--help"], ["check", "--help"], ["migrate", "--help"]])
def test_korean_help_has_no_english_argparse_boilerplate(argv, capsys):
    assert main(["--lang", "ko", *argv], env={}) == 0
    out = capsys.readouterr().out
    for phrase in ("usage:", "positional arguments", "options:", "show this help message", "show program"):
        assert phrase not in out
    assert "사용법:" in out and "옵션:" in out
    assert "사용법: 사용법" not in out and "usage: usage" not in out


def test_english_help_keeps_standard_headings(capsys):
    assert main(["--lang", "en", "--help"], env={}) == 0
    out = capsys.readouterr().out
    assert "usage:" in out and "options:" in out and "-h, --help" in out


# --- argparse usage errors follow the language too (round 3, Q-8) ---------------------------

USAGE_ERROR_CASES = {
    # name: (argv after --lang, English phrase, Korean phrase)
    "missing_required": (["check"], "the following arguments are required: --store, --embedder",
                         "다음 인자가 꼭 필요합니다: --store, --embedder"),
    "missing_subcommand": ([], "the following arguments are required:", "다음 인자가 꼭 필요합니다:"),
    "unrecognized": (["demo", "--bogus"], "unrecognized arguments: --bogus", "알 수 없는 인자: --bogus"),
    "invalid_int": (["migrate", "--store", "local://idx", "--to", "fake:b", "-k", "abc"],
                    "argument -k: invalid int value: 'abc'", "인자 -k: int 값이 올바르지 않습니다: 'abc'"),
    "invalid_float": (["migrate", "--store", "local://idx", "--to", "fake:b", "--min-self-recall", "x"],
                      "invalid float value: 'x'", "float 값이 올바르지 않습니다: 'x'"),
    "invalid_choice": (["bogus"], "invalid choice: 'bogus'", "고를 수 없는 값: 'bogus'"),
    "expected_one": (["check", "--store"], "argument --store: expected one argument",
                     "인자 --store: 값이 하나 있어야 합니다"),
    "ambiguous": (["migrate", "--s", "x", "--to", "fake:b"], "ambiguous option: --s could match",
                  "모호한 옵션: --s"),
    "not_migratable": (["migrate", "--store", "sidecar://x.json", "--to", "fake:b"],
                       "this store cannot be migrated", "이 저장소는 마이그레이션할 수 없습니다"),
}
ARGPARSE_ENGLISH = ("usage:", "error:", "required", "unrecognized", "invalid", "expected", "ambiguous",
                    "choose from", "argument ", "cannot be migrated")


@pytest.mark.parametrize("lang", ["en", "ko"])
@pytest.mark.parametrize("case", sorted(USAGE_ERROR_CASES))
def test_usage_errors_follow_the_language(case, lang, capsys):
    argv, english, korean = USAGE_ERROR_CASES[case]
    assert main(["--lang", lang, *argv], env={}) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    err = captured.err
    if lang == "ko":
        assert korean in err
        assert "오류:" in err
        for phrase in ARGPARSE_ENGLISH:
            assert phrase not in err, (phrase, err)
    else:
        assert english in err
        assert "error:" in err
        assert not HANGUL.search(err)


def test_usage_errors_follow_the_environment_language(capsys):
    assert main(["check"], env={"LANG": "ko_KR.UTF-8"}) == 2
    assert "다음 인자가 꼭 필요합니다" in capsys.readouterr().err
    assert main(["check", "--lang", "en"], env={"LANG": "ko_KR.UTF-8"}) == 2
    assert "the following arguments are required" in capsys.readouterr().err


def test_argparse_message_table_matches_this_python():
    """Every msgid we translate still exists in the running argparse, and both languages cover it."""
    import argparse
    import inspect

    from embedstamp.cli import _ARGPARSE_MSGIDS

    source = inspect.getsource(argparse)
    en, ko = load("en"), load("ko")
    placeholders = re.compile(r"%(?:\(\w+\))?[sr]")
    for msgid, key in _ARGPARSE_MSGIDS.items():
        assert repr(msgid)[1:-1] in source or msgid in source, msgid
        assert en[key] == msgid, key  # English table is the argparse original
        assert sorted(placeholders.findall(ko[key])) == sorted(placeholders.findall(msgid)), key


def test_argparse_translation_is_restored_after_main(capsys):
    import argparse

    before = (argparse._, argparse.ngettext)
    main(["--lang", "ko", "check"], env={})
    capsys.readouterr()
    assert (argparse._, argparse.ngettext) == before
    parser = argparse.ArgumentParser(prog="other")
    parser.add_argument("--n", type=int)
    with pytest.raises(SystemExit):
        parser.parse_args(["--n", "x"])
    assert "invalid int value" in capsys.readouterr().err
