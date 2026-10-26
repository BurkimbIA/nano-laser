# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- `LaserModelDownloader` creates a `model_dir` that does not exist yet, not only the default cache, and a download now gives up after 60 seconds without data instead of hanging.
- `encode_sentences([])` returns an empty `(0, dim)` array instead of raising `ValueError`.
- On PyPI, the author line read "Facebook AI Research" with a link to a maintainer's address. The three authors are now listed by name, and the maintainers' contact addresses are listed separately.

### Added

- A `dyu_Latn` case in the parity tests, and `LASER_PARITY_REQUIRED=1` to download missing models instead of skipping.

## v0.1.1 - 2026-10-26

### Fixed

- README links to the notebook, the parity results and the license now work on PyPI.
- The README documents installing from PyPI (`pip install nano-laser`, `uv add nano-laser`).

### Added

- Project URLs (repository, changelog, issues) in the package metadata, shown on PyPI.

## v0.1.0 - 2026-10-26

### Added

- `nano_laser`: LASER2 and LASER3 encoders in plain torch, a drop-in replacement for `laser_encoders` 0.0.2 without fairseq.
- Parity tests against embeddings from `laser_encoders`, run when the models are cached locally.
- A Colab notebook comparing `nano_laser` with `laser_encoders` on all 147 LASER3 models; all match.
- A workflow that publishes to PyPI with trusted publishing when a `vX.Y.Z` tag is pushed.

### Security

- Bumped the `uv-pre-commit` hook to uv 0.13.4. Earlier versions ship the compromised emSign ECC Root CA - G3 root, which certifi 2026.10.19 and uv 0.13.4 remove.
