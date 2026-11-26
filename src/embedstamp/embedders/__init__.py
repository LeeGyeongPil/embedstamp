"""Embedder adapters and the ``resolve_embedder`` spec parser."""

from __future__ import annotations

from urllib.parse import parse_qs

from .._vectors import Embedder
from .fake import FakeEmbedder

__all__ = ["Embedder", "FakeEmbedder", "resolve_embedder"]


def resolve_embedder(spec: str) -> tuple[Embedder, str]:
    """Turn a spec string into ``(embedder, model_name)``.

    * ``fake:<name>[?noise=0.05&scale=1.0&prefix=...&dim=384]`` - practice mock
    * ``fastembed:<model>`` - real model, e.g. ``fastembed:BAAI/bge-small-en-v1.5``
    """
    kind, sep, rest = spec.partition(":")
    if not sep or not rest:
        raise ValueError(
            f"invalid embedder spec {spec!r}; use fake:<name>[?noise=..] or fastembed:<model>"
        )
    if kind == "fake":
        name, _, query = rest.partition("?")
        opts = {k: v[-1] for k, v in parse_qs(query, keep_blank_values=True).items()}
        unknown = set(opts) - {"noise", "scale", "prefix", "dim"}
        if not name or unknown:
            raise ValueError(
                f"invalid fake spec {spec!r}; options are noise, scale, prefix, dim"
                + (f" (unknown: {', '.join(sorted(unknown))})" if unknown else "")
            )
        try:
            emb = FakeEmbedder(
                f"fake:{name}",
                dim=int(opts.get("dim", 384)),
                noise=float(opts.get("noise", 0.0)),
                scale=float(opts.get("scale", 1.0)),
                prefix=opts.get("prefix", ""),
            )
        except ValueError as exc:
            raise ValueError(f"invalid fake spec {spec!r}: {exc}") from exc
        return emb, f"fake:{name}"
    if kind == "fastembed":
        from .fastembed import FastEmbedEmbedder

        return FastEmbedEmbedder(rest), rest
    raise ValueError(f"unknown embedder kind {kind!r}; use fake or fastembed")
