# Daily Work Log

## 2026-08-23
- Added an initial Python package scaffold for Character Consistency Lab.
- Built `ccl-manifest`, a CLI that expands a TOML character-consistency spec into a reproducible JSON prompt manifest.
- Added deterministic per-sample seed derivation, an example experiment spec, and unit tests covering manifest expansion and stability.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests` and a sample manifest build.
