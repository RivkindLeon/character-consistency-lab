# Daily Work Log

## 2026-09-05
- Completed Milestone 1 with a versioned, typed benchmark schema and fixed 20-scene set.
- Added explicit reusable prompts and seeds covering every benchmark category in the implementation brief, including multi-character and complex compositions.
- Added validation for malformed scenes, duplicate IDs, invalid seeds, and unknown fields; updated README and `docs/PROGRESS.md` to make Milestone 2 the next work.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests -v` (32 tests passing), `./venv/bin/character-lab dataset --help`, and `git diff --check`.

## 2026-09-04
- Implemented Milestone 1 dataset statistics and the `character-lab dataset stats` command.
- Added deterministic per-character counts for train, validation, and reference splits plus source-image resolution distributions; invalid datasets are rejected before statistics are computed.
- Updated `docs/PROGRESS.md`; the fixed benchmark scene set is the next unfinished Milestone 1 item.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests -v` (27 tests passing), `./venv/bin/character-lab dataset --help`, and `git diff --check`.

## 2026-09-03
- Implemented the Milestone 1 dataset validator and the `character-lab dataset validate` command.
- Added conventional `characters.yaml` + `manifest.jsonl` loading and checks for missing/corrupt images, duplicate paths, dataset-root escapes, and content-based train/reference leakage without modifying source images.
- Added Pillow as the first image-processing dependency, documented the workflow, and kept `docs/PROGRESS.md` explicit that dataset stats and benchmark scenes remain unfinished.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests -v` (25 tests passing), `./venv/bin/character-lab --help`, and `git diff --check`.

## 2026-09-02
- Replaced ad-hoc TOML experiment parsing with strict typed YAML configuration using Pydantic and PyYAML.
- Added range checks, unknown-field rejection, friendly malformed-YAML CLI errors, a migrated example config, and updated documentation.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests -v` (23 tests passing), both CLI commands against the YAML example (128 prompts), and `git diff --check`.

## 2026-09-01
- Added the first real dataset abstraction: typed character definitions, dataset records, supported splits, and complete manifests independent of model code.
- Added JSONL manifest loading with actionable schema errors for invalid records, duplicate character metadata, unknown character IDs, and malformed JSON.
- Kept image access out of the loader so later validation can inspect files without silently modifying source data.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests -v` (22 tests passing) and `git diff --check`.

## 2026-08-31
- Added a backend-neutral model interface with validated generation requests and reproducible result metadata.
- Added a CPU-safe dry-run backend that never loads model libraries or writes an image, establishing the seam for future FLUX and SDXL implementations.
- Added unit coverage for request validation, backend lifecycle, dry-run metadata, and the no-artifact guarantee.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests -v`.

## 2026-08-30
- Added deterministic paired seeds for prompt-locked comparison groups, ensuring every model, LoRA, and hyperparameter variant for a scene starts from the same noise seed.
- Kept seeds distinct across scene groups and stable across repeated manifest builds.
- Added regression coverage for within-group seed pairing and cross-scene seed diversity, and documented the controlled-comparison behavior.

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

## 2026-08-26
- Added manifest-level comparison grouping so prompt-locked scene variants now carry a `comparison_group_id` and top-level `comparison_groups` metadata for downstream model/LoRA identity comparisons.
- Kept group membership stable across render sweeps while preserving existing per-sample render settings and sample IDs.
- Updated README output docs for comparison groups and verified the generated Mira manifest exposes 8 prompt groups over 128 samples.
- Verified with `PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests` and `./venv/bin/ccl-manifest build-manifest --spec examples/mira_consistency.toml --output out/mira_consistency.json`.
