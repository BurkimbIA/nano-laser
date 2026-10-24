# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Security

- Bumped the `uv-pre-commit` hook to uv 0.13.4. Earlier versions ship the compromised emSign ECC Root CA - G3 root, which certifi 2026.10.19 and uv 0.13.4 remove.
