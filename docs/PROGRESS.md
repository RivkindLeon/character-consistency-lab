# Progress

Milestone tracking against `docs/IMPLEMENTATION_BRIEF.md`. The nightly
development job reads this file to decide what to work on next.

**Keep this file honest.** It is the only memory that survives between nightly
sessions — each run starts with no history of previous runs. If it claims work
that was not done, the next session builds on a lie.

Last verified against the code: 2026-09-05.

---

## Milestone 0 — ML Project Foundation

**Status: complete.**

| Required | State |
|---|---|
| Python package | done — `src/character_consistency_lab/` |
| CLI | done — `ccl-manifest` (brief specifies `character-lab`) |
| Tests | done — 23 unit tests, all passing |
| Configuration system | done — strict typed YAML loading with Pydantic, including range and unknown-field validation |
| Dataset abstraction | done — typed character, record, split, and manifest contracts with JSONL loading |
| Model backend interface | done — backend-neutral generation request/result contracts and a CPU-safe dry-run backend |

The package layout still diverges from section 5 of the brief. The brief
specifies `src/character_lab/` with `data/`, `models/`, `training/`,
`inference/`, `evaluation/`, `experiments/`, `reports/`. The repository now has
the `data/` and `models/` boundaries, while the remaining requested package
boundaries are not yet present.

## Milestone 0.5 — Continuous Integration

**Status: complete.**

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`: installs the package and runs the full unit test suite. It downloads no
model weights, so it stays inside the constraint in section 28 of the brief.

Registered as a required status check on `main`. A red build now blocks the
merge instead of depending on the session's own judgement.

Not yet included: linting, formatting, type checking, and the `pytest -m gpu`
split section 28 asks for — the suite currently uses `unittest`. Add those when
the tools are actually introduced, not before.

**Before merging anything, read "If CI fails" in the brief.**

## Milestone 1 — Dataset + Benchmark

**Status: complete.**

What exists is a *prompt* manifest generator: it expands a typed YAML experiment config
into a JSON grid of prompt/seed/render-setting combinations, with deterministic
seeds and comparison groups. That is genuinely useful and it works, but it is
not what this milestone asks for.

The brief asks for a *dataset* manifest — images, character IDs, captions,
train/validation/reference splits — plus `character-lab dataset validate` and
`character-lab dataset stats`, plus a benchmark scene set of ~20 scenes in
`benchmarks/scenes.yaml`.

Dataset validation is now implemented and verified. The conventional dataset
layout uses `characters.yaml` plus `manifest.jsonl`. The validator reports
missing and corrupt images, duplicate manifest paths, schema errors including
missing captions and unknown characters, and train/reference leakage detected
from file content rather than filenames. It never modifies source images.

Dataset stats are implemented and verified through `character-lab dataset
stats`. The command validates the dataset first, then reports deterministic
per-character counts for every split and the source-image resolution
distribution without modifying images.

The versioned benchmark schema and fixed 20-scene set are implemented in
`benchmarks/scenes.yaml`. Each scene stores an explicit prompt and seed for
reuse across experiment variants. The set covers neutral, unusual-pose,
close-up, distant, lighting, background, camera-angle, two-character,
three-character, and complex-composition cases. Schema validation rejects
malformed scenes, duplicate IDs, invalid seeds, and unknown fields.

## Milestone 2 — Baseline Inference

**Status: not started.** No concrete model is ever loaded and nothing in this
repository generates an image. The only runtime dependencies are Pydantic and
PyYAML for configuration; no diffusion or tensor libraries are installed yet.

## Milestone 3 — LoRA Training

**Status: not started.**

## Milestone 4 — Evaluation

**Status: not started.** No identity metric, style metric, prompt alignment or
human review format.

## Milestones 5–8

**Status: not started.** Hyperparameter experiments, reference conditioning,
structural control, multi-character.

---

## Where the next session should start

Milestones 0, 0.5, and 1 are complete. Start Milestone 2 with baseline inference:
define a concrete, optional Diffusers backend that can be configured and dry-run
on this CPU-only host without importing or downloading model weights. Preserve
the backend-neutral interface and keep ordinary tests CPU-safe.

The dataset abstraction, manifest record schema, character metadata YAML
loading, and filesystem/image validation from sections 6–7 are implemented.
Dataset stats and the fixed benchmark scene set are complete.

The model backend interface is complete and verified. Concrete FLUX/SDXL
backends remain later implementation work under baseline inference.

## Hardware note

This server has no GPU and under 2 GB of RAM, so training and inference cannot
run here. That is **not** a reason to defer these milestones. Section 3 of the
brief explicitly requires the pipeline to still prepare datasets, validate
configuration, run unit tests, emit dry-run training commands, and support
execution on a remote GPU. Build behind clean interfaces and mock the expensive
layer, exactly as section 28 requires.
