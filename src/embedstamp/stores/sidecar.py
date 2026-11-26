"""SidecarStore: keep the fingerprint in a JSON file next to any index."""

from __future__ import annotations

import os
from pathlib import Path

from ..fingerprint import Fingerprint


def atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` without leaving a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class SidecarStore:
    """Fingerprint in a standalone JSON file, e.g. ``index.embedstamp.json``."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def load(self) -> Fingerprint | None:
        if not self.path.exists():
            return None
        return Fingerprint.from_file(self.path)

    def save(self, fp: Fingerprint) -> None:
        atomic_write(self.path, fp.to_json())
