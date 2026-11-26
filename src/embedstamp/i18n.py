"""Language choice and message lookup for the CLI (English and Korean)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib import resources

SUPPORTED = ("en", "ko")
ENV_ORDER = ("EMBEDSTAMP_LANG", "LC_ALL", "LC_MESSAGES", "LANG")


def _normalise(value: str) -> str:
    """Map a language or locale string such as ``ko_KR.UTF-8`` to ``ko`` or ``en``."""
    return "ko" if value.strip().lower().startswith("ko") else "en"


def resolve_lang(cli_lang: str | None, env: Mapping[str, str]) -> str:
    """Pick the language: ``--lang`` > EMBEDSTAMP_LANG > LC_ALL > LC_MESSAGES > LANG.

    The first non-empty source wins; anything that does not start with ``ko`` means English.
    """
    if cli_lang:
        return _normalise(cli_lang)
    for var in ENV_ORDER:
        value = env.get(var, "")
        if value.strip():
            return _normalise(value)
    return "en"


@lru_cache(maxsize=4)
def load_messages(lang: str) -> dict[str, str]:
    """Load ``locales/<lang>.json``."""
    if lang not in SUPPORTED:
        raise ValueError(f"unsupported language {lang!r}; choose one of {', '.join(SUPPORTED)}")
    text = resources.files("embedstamp").joinpath("locales", f"{lang}.json").read_text(encoding="utf-8")
    return json.loads(text)


class Translator:
    """``tr("key", name=value)`` returns the message of the chosen language."""

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self._messages = load_messages(lang)

    def __call__(self, key: str, **fields: object) -> str:
        return self._messages[key].format(**fields)

    def raw(self, key: str) -> str:
        """The message text without ``str.format`` (for %-style argparse templates)."""
        return self._messages[key]
