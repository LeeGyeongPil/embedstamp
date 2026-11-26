# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.2] - 2026-11-27

### Changed

- The missing-extra `ImportError` now has one message everywhere (`from embedstamp import ...`,
  `ChromaStore.open()`, `chroma://` stores, `FastEmbedEmbedder`, `fastembed:` specs), naming both
  `pip install "embedstamp[<extra>]"` and `uv sync --extra <extra>` for a clone.
- CI: named steps, a version report step, a 20-minute job timeout, read-only token permissions,
  and failed tests shown as annotations on the run page.
- CI: the required `core` job (push, pull request, manual) installs only the core dependency and
  pytest and runs `pytest -m "not timing"`, the demo and the quickstart. The Chroma tests moved to
  an `extras` job that runs only when started by hand. The weekly scheduled run was removed.
- Development: `chromadb` is no longer in the `dev` dependency group, so `uv sync` installs the
  core only. Run the full suite with `uv run --extra chroma pytest -q`. Chroma tests carry a
  `chroma` marker and are skipped without the extra; the demo's 10-second limit is a `timing`
  marker test. Library code changed only in the ImportError wording (four source files: `errors.py`, `__init__.py`, `stores/chroma.py`, `embedders/fastembed.py`).
- Docs: the README development section gives the core and full test commands without test
  counts (a test keeps it that way), the Chroma example runs with `uv run --extra chroma`, and the
  troubleshooting table covers `No module named 'chromadb'` in a clone.

## [0.1.1] - 2026-11-27

### Changed

- CI: update GitHub Actions to Node 24 runtimes and pin the uv version. No library changes.

## [0.1.0] - 2026-11-27

### Added

- `Fingerprint`, `stamp()`, `verify()`, `Thresholds`, `Verdict`: record an embedding model's
  identity (name, dimension, dtype, normalisation, norm, prefixes, fixed probe-sentence vectors)
  and judge a current embedder as `same`, `drift` or `different`.
- `Guard`: checks once (cached) before embedding documents or queries; raises
  `EmbeddingMismatchError` or warns.
- `migrate()`: blue/green re-embedding with resumable batches, self-recall@k, top-k overlap and an
  explicit name switch.
- Stores: `SidecarStore`, `LocalIndex`, `ChromaStore` (extra `chroma`).
- Embedders: `FakeEmbedder` (practice mock), `FastEmbedEmbedder` (extra `fastembed`),
  `resolve_embedder()` spec strings.
- CLI `embedstamp` with `demo`, `stamp`, `check`, `show`, `migrate`; English and Korean output
  (`--lang`, `EMBEDSTAMP_LANG`, `LC_ALL`, `LC_MESSAGES`, `LANG`).
