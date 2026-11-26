"""Command line interface: demo, stamp, check, show, migrate (English and Korean)."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from . import __version__
from .demo import run_demo
from .embedders import resolve_embedder
from .errors import EmbedstampError, FingerprintTooLargeError, SwitchRefusedError
from .fingerprint import stamp
from .i18n import Translator, resolve_lang
from .migrate import migrate
from .stores import MigratableStore, resolve_store
from .verify import verify

EXIT_SAME, EXIT_ERROR, EXIT_USAGE, EXIT_SAVE_FAILED = 0, 1, 2, 4
EXIT_DRIFT, EXIT_DIFFERENT, EXIT_NO_FINGERPRINT, EXIT_SWITCH_REFUSED = 10, 11, 12, 13
_CHECK_EXIT = {"same": EXIT_SAME, "drift": EXIT_DRIFT, "different": EXIT_DIFFERENT}


def _prescan_lang(argv: Sequence[str]) -> str | None:
    """Find ``--lang`` before argparse runs, so that --help can already be localised."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


# argparse 가 사용자에게 보여 주는 영어 원문(msgid) -> 메시지 표의 키.
# argparse 는 모든 문구를 모듈 전역 gettext 함수 `_`·`ngettext` 로 감싸 만든다.
# 그래서 파싱하는 동안만 그 두 함수를 바꿔 끼우면 오류·사용법 문구가 한 곳(locales)에서 번역된다.
# 표에 없는 원문(개발자용 오류 등)은 영어 그대로 둔다.
_ARGPARSE_MSGIDS: dict[str, str] = {
    "%(prog)s: error: %(message)s\n": "argparse.error",
    "usage: ": "argparse.usage",
    "%(heading)s:": "argparse.heading",
    " (default: %(default)s)": "argparse.default",
    "subcommands": "argparse.subcommands",
    "argument %(argument_name)s: %(message)s": "argparse.argument",
    "the following arguments are required: %s": "argparse.required",
    "one of the arguments %s is required": "argparse.one_of_required",
    "unrecognized arguments: %s": "argparse.unrecognized",
    "invalid %(type)s value: %(value)r": "argparse.invalid_type",
    "invalid choice: %(value)r (choose from %(choices)s)": "argparse.invalid_choice",
    "unknown parser %(parser_name)r (choices: %(choices)s)": "argparse.unknown_parser",
    "expected one argument": "argparse.expected_one",
    "expected at most one argument": "argparse.expected_at_most_one",
    "expected at least one argument": "argparse.expected_at_least_one",
    "expected %s argument": "argparse.expected_n",
    "expected %s arguments": "argparse.expected_n_plural",
    "ambiguous option: %(option)s could match %(matches)s": "argparse.ambiguous",
    "not allowed with argument %s": "argparse.not_allowed_with",
    "ignored explicit argument %r": "argparse.ignored_explicit",
    "unexpected option string: %s": "argparse.unexpected_option",
    "can't open '%(filename)s': %(error)s": "argparse.cant_open",
}


@contextlib.contextmanager
def _argparse_language(tr: Translator) -> Iterator[None]:
    """While active, argparse builds its messages from the chosen language's table."""
    original_gettext, original_ngettext = argparse._, argparse.ngettext

    def gettext(msgid: str) -> str:
        key = _ARGPARSE_MSGIDS.get(msgid)
        return tr.raw(key) if key else msgid

    def ngettext(singular: str, plural: str, n: int) -> str:
        return gettext(singular if n == 1 else plural)

    argparse._ = gettext  # type: ignore[attr-defined]
    argparse.ngettext = ngettext  # type: ignore[attr-defined]
    try:
        yield
    finally:  # 라이브러리로 import 한 다른 코드의 argparse 에 영향이 남지 않게 반드시 되돌린다
        argparse._ = original_gettext  # type: ignore[attr-defined]
        argparse.ngettext = original_ngettext  # type: ignore[attr-defined]


def _formatter_for(tr: Translator) -> type[argparse.HelpFormatter]:
    """A help formatter whose "usage:" prefix is localised."""

    class _Formatter(argparse.HelpFormatter):
        def add_usage(self, usage: Any, actions: Any, groups: Any, prefix: str | None = None) -> None:
            if prefix is None:  # argparse passes "" itself when it builds sub-command prog names
                prefix = tr("help.usage") + ": "
            super().add_usage(usage, actions, groups, prefix=prefix)

    return _Formatter


def _localise(parser: argparse.ArgumentParser, tr: Translator) -> None:
    """Add a localised -h and rename argparse's built-in group titles."""
    parser.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help=tr("help.help"))
    parser._positionals.title = tr("help.positional")
    parser._optionals.title = tr("help.options")


def _build_parser(tr: Translator) -> argparse.ArgumentParser:
    fmt = _formatter_for(tr)
    parser = argparse.ArgumentParser(
        prog="embedstamp", description=tr("app.description"), epilog=tr("app.epilog"),
        add_help=False, formatter_class=fmt,
    )
    _localise(parser, tr)
    parser.add_argument("--version", action="version", version=f"embedstamp {__version__}",
                        help=tr("help.version"))
    parser.add_argument("--lang", metavar="en|ko", help=tr("opt.lang"))
    sub = parser.add_subparsers(dest="command", required=True, metavar="{demo,stamp,check,show,migrate}")

    def add(name: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=tr(f"cmd.{name}"), description=tr(f"cmd.{name}"),
                           add_help=False, formatter_class=fmt)
        _localise(p, tr)
        p.add_argument("--lang", metavar="en|ko", default=argparse.SUPPRESS, help=tr("opt.lang"))
        return p

    add("demo")

    p = add("stamp")
    p.add_argument("--store", required=True, help=tr("opt.store"))
    p.add_argument("--embedder", required=True, help=tr("opt.embedder"))
    p.add_argument("--model", help=tr("opt.model"))
    p.add_argument("--doc-prefix", default="", help=tr("opt.doc_prefix"))
    p.add_argument("--query-prefix", default="", help=tr("opt.query_prefix"))
    p.add_argument("--compact", action="store_true", help=tr("opt.compact"))

    p = add("check")
    p.add_argument("--store", required=True, help=tr("opt.store"))
    p.add_argument("--embedder", required=True, help=tr("opt.embedder"))
    p.add_argument("--model", help=tr("opt.model"))
    p.add_argument("--doc-prefix", default="", help=tr("opt.doc_prefix"))
    p.add_argument("--query-prefix", default="", help=tr("opt.query_prefix"))
    p.add_argument("--json", action="store_true", help=tr("opt.json"))

    p = add("show")
    p.add_argument("--store", required=True, help=tr("opt.store"))
    p.add_argument("--json", action="store_true", help=tr("opt.json"))

    p = add("migrate")
    p.add_argument("--store", required=True, help=tr("opt.store"))
    p.add_argument("--to", required=True, dest="to", help=tr("opt.to"))
    p.add_argument("--from", dest="from_", help=tr("opt.from"))
    p.add_argument("--queries", help=tr("opt.queries"))
    p.add_argument("-k", type=int, default=10, help=tr("opt.k"))
    p.add_argument("--batch-size", type=int, default=64, help=tr("opt.batch_size"))
    p.add_argument("--doc-prefix", default="", help=tr("opt.doc_prefix"))
    p.add_argument("--query-prefix", default="", help=tr("opt.query_prefix"))
    p.add_argument("--switch", action="store_true", help=tr("opt.switch"))
    p.add_argument("--min-self-recall", type=float, default=0.9, help=tr("opt.min_self_recall"))
    return parser


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _error(text: str) -> None:
    """Print to stderr after flushing stdout, so lines appear in the order they happened.

    stdout 은 파이프·파일로 보낼 때 버퍼에 쌓이고 stderr 는 바로 나가서, 그대로 두면
    오류가 그보다 먼저 찍은 진행 줄보다 앞에 보인다.
    """
    sys.stdout.flush()
    print(text, file=sys.stderr, flush=True)


def _cmd_stamp(args: argparse.Namespace, tr: Translator) -> int:
    store = resolve_store(args.store)
    embed, spec_model = resolve_embedder(args.embedder)
    fp = stamp(embed, model=args.model or spec_model, doc_prefix=args.doc_prefix,
               query_prefix=args.query_prefix, compact=args.compact)
    try:
        store.save(fp)
    except (OSError, FingerprintTooLargeError) as exc:
        _error(tr("err.generic", message=exc))
        return EXIT_SAVE_FAILED
    print(tr("stamp.done", store=args.store, model=fp.model, dim=fp.dim, digest=fp.digest[:12],
             size=len(fp.to_json().encode("utf-8"))))
    return EXIT_SAME


def _cmd_check(args: argparse.Namespace, tr: Translator) -> int:
    store = resolve_store(args.store)
    embed, spec_model = resolve_embedder(args.embedder)  # validate the spec before touching the store
    stored = store.load()
    if stored is None:
        if args.json:
            print(json.dumps({"status": "missing", "min_cos": None, "reasons": ["fingerprint_not_found"]}))
        else:
            _error(tr("check.nofp", store=args.store))
        return EXIT_NO_FINGERPRINT
    verdict = verify(stored, embed, model=args.model or spec_model, doc_prefix=args.doc_prefix,
                     query_prefix=args.query_prefix)
    if args.json:
        print(json.dumps({
            "status": verdict.status, "min_cos": verdict.min_cos, "mean_cos": verdict.mean_cos,
            "reasons": list(verdict.reasons), "stored_model": verdict.stored.model,
            "current_model": verdict.current.model,
        }))
    else:
        print(tr("check.status", status=verdict.status, min_cos=_fmt(verdict.min_cos),
                 mean_cos=_fmt(verdict.mean_cos)))
        print(tr("check.models", stored=verdict.stored.model, current=verdict.current.model))
        if verdict.reasons:
            print(tr("check.reasons"))
            for code in verdict.reasons:
                print(f"  - {code}: {tr('reason.' + code)}")
        print(tr(f"check.hint.{verdict.status}"))
    return _CHECK_EXIT[verdict.status]


def _cmd_show(args: argparse.Namespace, tr: Translator) -> int:
    store = resolve_store(args.store)
    fp = store.load()
    if fp is None:
        _error(tr("check.nofp", store=args.store))
        return EXIT_NO_FINGERPRINT
    if args.json:
        data: dict[str, Any] = dataclasses.asdict(fp)
        data["probes_fp16"] = fp.probes_fp16 is not None  # the raw payload is too long to print
        data.pop("projection32")
        print(json.dumps(data, ensure_ascii=False))
        return EXIT_SAME
    print(tr("show.title", store=args.store))
    print(tr("show.model", model=fp.model))
    print(tr("show.dim", dim=fp.dim, dtype=fp.dtype))
    print(tr("show.norm", normalized=fp.normalized, norm_mean=fp.norm_mean))
    print(tr("show.prefixes", doc_prefix=fp.doc_prefix, query_prefix=fp.query_prefix))
    print(tr("show.probes", probe_set=fp.probe_set, n_probes=fp.n_probes,
             mode="full" if fp.probes_fp16 else "compact"))
    print(tr("show.digest", digest=fp.digest))
    print(tr("show.created", created_at=fp.created_at))
    return EXIT_SAME


def _cmd_migrate(args: argparse.Namespace, tr: Translator) -> int:
    store = resolve_store(args.store)
    if not isinstance(store, MigratableStore):
        _error(tr("err.generic", message=tr("err.not_migratable")))
        return EXIT_USAGE
    new_embed, new_model = resolve_embedder(args.to)
    old_embed = resolve_embedder(args.from_)[0] if args.from_ else None
    queries = None
    if args.queries:
        with open(args.queries, encoding="utf-8") as fh:
            queries = [line.strip() for line in fh if line.strip()]
    # 문서가 없는 인덱스(예: `stamp` 만 한 local://)는 "진행 중" 줄을 찍기 전에 대처법과 함께 멈춘다
    if store.count() == 0:
        _error(tr("err.generic", message=tr("err.empty_source", store=args.store)))
        return EXIT_ERROR
    print(tr("migrate.start", store=args.store, model=new_model), flush=True)
    try:
        report = migrate(
            store, new_embed, new_model=new_model, old_embed=old_embed, queries=queries, k=args.k,
            batch_size=args.batch_size, switch=args.switch, doc_prefix=args.doc_prefix,
            query_prefix=args.query_prefix, min_self_recall=args.min_self_recall,
            progress=lambda done, total: print(tr("migrate.progress", done=done, total=total), flush=True),
        )
    except SwitchRefusedError as exc:
        _error(tr("migrate.refused", message=exc))
        return EXIT_SWITCH_REFUSED
    # green= 에는 green 컬렉션을 만든 이름(예: docs__green)을 보여 준다. 전환 뒤의 이름은 아래 switched 줄에 있다.
    print(tr("migrate.report", n_docs=report.n_docs, n_embedded=report.n_embedded,
             n_skipped=report.n_skipped, green=report.built_name or report.green_name))
    if report.n_embedded == 0 and report.n_skipped > 0:  # 이전 실행에서 이미 다 채운 green 을 다시 쓴 경우
        print(tr("migrate.reused", green=report.built_name or report.green_name))
    print(tr("migrate.self_recall", k=report.k, value=f"{report.self_recall_at_k:.2f}"))
    if report.overlap_at_k is not None:
        print(tr("migrate.overlap", k=report.k, value=f"{report.overlap_at_k:.2f}"))
    if report.switched:
        print(tr("migrate.switched", store=report.green_name, model=new_model, blue=report.blue_name))
        after = report.verdict_after_switch
        if after is not None:  # 전환 직후 실제로 다시 잰 수치를 보여 준다
            print(tr("check.status", status=after.status, min_cos=_fmt(after.min_cos),
                     mean_cos=_fmt(after.mean_cos)))
    else:
        print(tr("migrate.not_switched", green=report.green_name))
    return EXIT_SAME


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    environment = os.environ if env is None else env
    tr = Translator(resolve_lang(_prescan_lang(args_list), environment))
    try:
        with _argparse_language(tr):  # 도움말·사용법 오류 문구도 고른 언어로
            parser = _build_parser(tr)
            args = parser.parse_args(args_list)
    except SystemExit as exc:  # --help, --version and usage errors
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    try:
        if args.command == "demo":
            return run_demo(tr)
        handler = {"stamp": _cmd_stamp, "check": _cmd_check, "show": _cmd_show, "migrate": _cmd_migrate}
        return handler[args.command](args, tr)
    except ValueError as exc:
        _error(tr("err.generic", message=exc))
        return EXIT_USAGE
    except (EmbedstampError, ImportError, OSError) as exc:
        _error(tr("err.generic", message=exc))
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
