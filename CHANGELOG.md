# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
