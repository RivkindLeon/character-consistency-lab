# Daily Work Log

## 2026-08-28
- Added structural validation for experiment specs, including required experiment metadata, non-empty variant/sweep arrays, typed render values, and positive dimensions/step counts.
- Added `ccl-manifest validate-spec` with concise failures for invalid specs, malformed TOML, and unreadable files instead of Python tracebacks.
- Updated the README workflow and expanded the test suite to cover validation behavior and CLI success/failure paths.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests -v`, `./venv/bin/ccl-manifest validate-spec --spec examples/mira_consistency.toml`, and a 128-sample example manifest build.

## 2026-08-25
- Extended `ccl-manifest` to support checkpoint-level render sweeps via `model_ids` and `lora_adapters`, so character-consistency runs can compare base models and adapter revisions in one manifest.
- Propagated `model_id` and `lora_adapter` into per-sample `render_settings` and sample IDs for downstream Diffusers runners and traceable experiment outputs.
- Updated the example spec and README to document model/adaptor sweeps alongside existing prompt and hyperparameter grids.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests` and `./venv/bin/ccl-manifest build-manifest --spec examples/mira_consistency.toml --output out/mira_consistency.json`.

## 2026-08-24
- Extended `ccl-manifest` to support render-parameter sweeps for guidance scale, LoRA strength, inference steps, and canvas size.
- Emitted per-sample `render_settings` metadata so downstream Diffusers runners can execute identity experiments without re-parsing the TOML spec.
- Updated the example spec and README to show prompt variation plus render sweeps in one reproducible manifest.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests` and `./venv/bin/ccl-manifest build-manifest --spec examples/mira_consistency.toml --output out/mira_consistency.json` after `./venv/bin/pip install -e .`.

## 2026-08-23
- Added an initial Python package scaffold for Character Consistency Lab.
- Built `ccl-manifest`, a CLI that expands a TOML character-consistency spec into a reproducible JSON prompt manifest.
- Added deterministic per-sample seed derivation, an example experiment spec, and unit tests covering manifest expansion and stability.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests` and a sample manifest build.
