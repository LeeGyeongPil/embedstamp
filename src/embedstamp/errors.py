"""Exception types raised by embedstamp. Messages are English and say how to fix the problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .verify import Verdict


class EmbedstampError(Exception):
    """Base class for every error embedstamp raises on purpose."""


class EmbeddingMismatchError(EmbedstampError):
    """The current embedder does not produce the vectors the index was built with."""

    def __init__(self, verdict: "Verdict") -> None:
        self.verdict = verdict
        cos = "n/a" if verdict.min_cos is None else f"{verdict.min_cos:.4f}"
        super().__init__(
            f"embedding model mismatch: status={verdict.status}, min_cos={cos}, "
            f"reasons={', '.join(verdict.reasons) or '-'}; "
            f"stored model={verdict.stored.model!r}, current model={verdict.current.model!r}. "
            "Use the embedder the index was built with, or re-embed with embedstamp.migrate()."
        )


class FingerprintNotFoundError(EmbedstampError):
    """The store has no fingerprint yet."""

    def __init__(self, where: str = "the store") -> None:
        super().__init__(
            f"no fingerprint found in {where}; run embedstamp.stamp(...) and store.save(...) "
            "(or `embedstamp stamp`) while the index is still built with a known model."
        )


class FingerprintTooLargeError(EmbedstampError):
    """The fingerprint does not fit the store's metadata limit."""


class SwitchRefusedError(EmbedstampError):
    """migrate(switch=True) was asked to switch, but the new collection failed its quality check.

    The finished report (green collection kept, nothing switched) is available as ``.report``.
    """

    def __init__(self, message: str, report: object | None = None) -> None:
        super().__init__(message)
        self.report = report
