# VGSP 0.1.1 — PyPI documentation refresh

This is a documentation and packaging-metadata release. The VGSP algorithm, binary supervised API, and default `n_features=7` remain unchanged.

## Changes
- Redesigned the PyPI description with a clear Quick Start, Features table, Python API, Pipeline and StratifiedKFold examples, method explanation, testing, limitations, and citation.
- Fixed the install command throughout: `pip install vgsp` (not `vgsp-ml`).
- Added an explicit distinction between the existing research notebook DOI and packaged software releases.
- Updated the version to 0.1.1.

## Publishing steps

1. Commit these three files to `nqmn/vgsp` on the `main` branch **without deleting any other files**:
   - `PYPI_README.md`
   - `pyproject.toml`
   - `src/vgsp/__init__.py`
2. Verify that the PyPI Trusted Publisher is configured for **project `vgsp`**, GitHub owner `nqmn`, repo `vgsp`, workflow `publish-pypi.yml`, environment `pypi`.
3. Run GitHub Actions -> Publish to PyPI -> Run workflow on `main`, if no release-triggered publish is desired. Otherwise create a GitHub release with tag **`v0.1.1`** pointing at the updated `main`, and the release event should invoke the workflow.
4. Review the run logs and confirm https://pypi.org/project/vgsp/ shows version 0.1.1 and the updated description.

**Important:** Existing PyPI versions are immutable. Re-uploading 0.1.0 will fail even if only README text changed. If version 0.1.1 is already taken, choose the next *unused* version and keep the version in `pyproject.toml`, `src/vgsp/__init__.py`, and release notes consistent.

The public repository's research README and notebook are intentionally left unchanged. No PyPI release has been published by preparation of these files alone.
