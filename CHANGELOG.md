# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- `nano_laser`: LASER2 and LASER3 encoders in plain torch, a drop-in replacement for `laser_encoders` 0.0.2 without fairseq.
- Parity tests against embeddings from `laser_encoders`, run when the models are cached locally.
- A Colab notebook comparing `nano_laser` with `laser_encoders` on all 147 LASER3 models; all match.
- A workflow that publishes to PyPI with trusted publishing when a `vX.Y.Z` tag is pushed.

### Security

- Bumped the `uv-pre-commit` hook to uv 0.13.4. Earlier versions ship the compromised emSign ECC Root CA - G3 root, which certifi 2026.10.19 and uv 0.13.4 remove.
