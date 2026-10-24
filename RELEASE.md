# Release process

This guide describes how to release `python-package-template`.

## Versioning

The project follows [Semantic Versioning](https://semver.org/):

- `MAJOR`: breaking API, configuration, or workflow changes.
- `MINOR`: backwards-compatible features.
- `PATCH`: backwards-compatible bug fixes, dependency fixes, and security patches.

The release version is the value of `project.version` in [`pyproject.toml`](pyproject.toml). The release tag and changelog heading must use the same version, formatted as `vX.Y.Z`.

## Prepare the release

Start from an up-to-date `main` branch and create a release branch:

```bash
git checkout main
git pull --ff-only origin main
git checkout -b chore/release-vX.Y.Z
```

Choose the appropriate version bump:

```bash
uv version --bump major
uv version --bump minor
uv version --bump patch
```

This updates `pyproject.toml` and `uv.lock`. Do not edit only one of them.

Update `CHANGELOG.md` as follows:

1. Move the release-ready entries from `Unreleased` into `## vX.Y.Z - YYYY-MM-DD`.
2. Add a new empty `## Unreleased` section directly below the changelog intro.
3. Review the entries for user-facing accuracy and remove internal-only detail that does not help users upgrade.

## Verify and open the release PR

Run the release checks:

```bash
uv lock --check
just format
just lint
just typecheck
just test
just pre-commit
```

Review the resulting diff. It should include the intended version change, lockfile update, and changelog update, with any relevant documentation or configuration changes.

```bash
git add pyproject.toml uv.lock CHANGELOG.md
git commit -m "chore: release vX.Y.Z"
git push -u origin chore/release-vX.Y.Z
```

Open a pull request targeting `main` and merge it after review and all enabled CI checks pass.

## Tag and publish

After the release PR is merged, tag the exact `main` commit with an annotated tag, then push it:

```bash
git checkout main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin vX.Y.Z
```

Create the GitHub Release using generated notes, then review and edit the notes before publishing:

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --generate-notes
```

## Post-release checks

- Confirm the Git tag points to the merged release commit.
- Confirm the GitHub Release title, version, and notes are correct.
- Confirm that `CHANGELOG.md` has an empty `Unreleased` section for the next change.

This repository does not currently contain a package publishing workflow (e.g. to PyPI). If one is added, document its trigger and verification steps here before relying on it for a release.
