"""Docs stay true as tests are added: no test counts in the READMEs, one version everywhere."""

import re
import tomllib
from pathlib import Path

import pytest

import embedstamp

ROOT = Path(__file__).resolve().parents[1]
READMES = ["README.md", "README.ko.md"]

# pytest 요약 형식("12 passed")과 "N개 테스트" — 테스트를 하나 더할 때마다 틀려지는 숫자.
# CLI 의 key=value 출력(`embedded=160 skipped=0`)은 실제 실행 결과라 대상이 아니다: '=' 바로 뒤 숫자는 건너뛴다.
COUNT_PATTERNS = [
    re.compile(r"(?<![=\d])\b\d+ (passed|failed|skipped|deselected|xfailed|xpassed|errors?)\b"),
    re.compile(r"\d+\s*개의?\s*테스트"),
]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def section(text: str, heading: str) -> str:
    """The body of a ``## heading`` section, up to the next ``## ``."""
    start = text.index(f"\n## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else None]


@pytest.mark.parametrize("name", READMES)
def test_readme_has_no_test_counts(name):
    text = read(name)
    found = [m.group(0) for pattern in COUNT_PATTERNS for m in pattern.finditer(text)]
    assert found == [], f"{name} mentions test counts: {found}"


def test_count_pattern_catches_summaries_but_not_cli_output():
    def hits(text):
        return [m.group(0) for pattern in COUNT_PATTERNS for m in pattern.finditer(text)]

    assert hits("121 passed, 1 skipped in 2.1s") == ["121 passed", "1 skipped"]
    assert hits("테스트 123개 중 123 개의 테스트") == ["123 개의 테스트"]
    assert hits("docs=30 embedded=0 skipped=30 green=kb__green") == []


@pytest.mark.parametrize(
    ("name", "heading", "expected"),
    [("README.md", "Development", "no failures"), ("README.ko.md", "개발·기여", "실패 없이")],
)
def test_development_section_gives_core_and_full_commands(name, heading, expected):
    body = section(read(name), heading)
    assert re.search(r"^uv run pytest -q(\s|$)", body, re.M)  # 코어(chromadb 없이)
    assert re.search(r"^uv run --extra chroma pytest -q(\s|$)", body, re.M)  # 전체
    assert expected in body  # 기대 결과는 개수 없이


def test_pyproject_version_matches_package_version():
    project = tomllib.loads(read("pyproject.toml"))["project"]
    assert project["version"] == embedstamp.__version__


def test_changelog_has_a_heading_for_the_current_version():
    headings = re.findall(r"^## \[([^\]]+)\]", read("CHANGELOG.md"), re.M)
    assert embedstamp.__version__ in headings
