# Character Consistency Lab

## Fine-Tuning and Evaluation of Diffusion Models for Recurring Character Generation

# 1. Project Goal

Build an ML experimentation pipeline for solving a specific generative-image problem:

> How can a diffusion model preserve the identity and visual style of recurring fictional characters across dozens of different scenes?

The original motivation is creating a long illustrated story/book where the same characters appear across ~50 images.

Current image generators can create strong individual illustrations but recurring characters often drift:

- face changes;
- body proportions change;
- colors change;
- character details disappear;
- style changes;
- multiple characters become mixed together;
- composition becomes difficult to control.

The purpose of this repository is to **experimentally measure and improve character consistency**.

This is NOT primarily a book-generation application.

This is an ML engineering / experimentation project.

---

# 2. Main Research Question

Compare several approaches:

```text
Base diffusion model
        vs
Character LoRA
        vs
Reference conditioning
        vs
LoRA + reference conditioning
        vs
LoRA + reference conditioning + structural control
```

Measure how each affects:

```text
character identity
style consistency
prompt adherence
composition
generation quality
```

---

# 3. Initial Model Strategy

Primary fine-tuning target:

```text
FLUX.2 Klein Base 4B
```

Do not couple the architecture permanently to one model.

Create a model backend abstraction so experiments can later use:

```text
FLUX
SDXL
other Diffusers-compatible models
```

If local hardware cannot practically execute a specific experiment, the pipeline must still:

- prepare datasets;
- validate configuration;
- run unit tests;
- produce dry-run training commands;
- support execution on a remote GPU.

Do not commit large model weights to Git.

---

# 4. Tech Stack

Core:

```text
Python
PyTorch
Hugging Face Diffusers
Transformers
PEFT
Accelerate
safetensors
Pillow
NumPy
pandas
```

CLI:

```text
Typer
```

Configuration:

```text
YAML + Pydantic
```

Evaluation:

```text
CLIP-compatible embeddings
DINO-style visual embeddings
cosine similarity
human evaluation
```

Visualization:

```text
matplotlib
HTML report generation
```

Optional UI comes later.

Do not start by building a React application.

The ML pipeline comes first.

---

# 5. Repository Structure

```text
character-consistency-lab/

src/
  character_lab/

    data/
      dataset.py
      preprocessing.py
      validation.py

    models/
      base.py
      flux.py
      sdxl.py

    training/
      lora.py
      configs.py

    inference/
      generate.py
      adapters.py

    evaluation/
      identity.py
      style.py
      prompt_alignment.py
      human_review.py
      aggregate.py

    experiments/
      runner.py
      registry.py

    reports/
      report.py

    cli.py

configs/
  models/
  training/
  experiments/

datasets/
  demo/

benchmarks/
  scenes.yaml

runs/

tests/

scripts/

docs/

README.md
pyproject.toml
```

Generated images and large run artifacts should generally be gitignored.

Commit only small curated examples needed for README/demo.

---

# 6. Dataset Format

Do not hardcode one character.

Create a general dataset schema.

Example:

```yaml
characters:
  - id: dino
    trigger: "chr_dino"

  - id: tira
    trigger: "chr_tira"

  - id: dipo
    trigger: "chr_dipo"
```

Manifest:

```json
{
  "image": "images/dino_001.png",
  "character": "dino",
  "caption": "chr_dino standing in a sunny meadow",
  "split": "train"
}
```

Support:

```text
train
validation
reference
```

Dataset validation should check:

- missing images;
- duplicate filenames;
- invalid dimensions;
- missing captions;
- unknown character IDs;
- train/reference leakage.

---

# 7. Dataset Preparation

Implement CLI:

```bash
character-lab dataset validate datasets/my_dataset
```

and:

```bash
character-lab dataset stats datasets/my_dataset
```

Output example:

```text
Characters: 3

Dino
training: 14
reference: 4

Tira
training: 12
reference: 4

Dipo
training: 13
reference: 4

Resolution distribution:
...
```

Dataset preparation must be reproducible.

Do not silently modify original images.

Processed images should go to a separate directory.

---

# 8. Benchmark Scene Set

Create a fixed benchmark independent from training data.

Example:

```yaml
- id: dino_portrait
  characters:
    - dino
  prompt: >
    chr_dino standing in a sunny meadow,
    children's book illustration

- id: dino_running
  characters:
    - dino
  prompt: >
    chr_dino running through tall grass

- id: dino_night
  characters:
    - dino
  prompt: >
    chr_dino looking at the moon at night
```

Benchmark categories:

```text
single character / neutral
single character / unusual pose
single character / close-up
single character / distant
lighting changes
different backgrounds
different camera angles

two characters
three characters

complex composition
```

Start with approximately 20 benchmark scenes.

Design the format to grow to ~50 scenes.

The same prompts and seeds must be reusable across experiments.

---

# 9. Reproducibility

Every generation run must save:

```text
model
model revision if available
LoRA
LoRA weight
prompt
negative prompt if used
seed
width
height
steps
guidance
adapter configuration
timestamp
git commit SHA when possible
```

Example:

```json
{
  "experiment": "dino_lora_v3",
  "seed": 42137,
  "model": "...",
  "lora": "...",
  "lora_scale": 0.8
}
```

A result without its configuration is not a valid experiment result.

---

# 10. Baseline Experiment

The first real experiment must use the base model with **no character fine-tuning**.

Generate the benchmark.

Store:

```text
runs/baseline/
```

This establishes how poorly or well the pretrained model handles the characters without adaptation.

Do not skip the baseline.

---

# 11. LoRA Training

Implement character LoRA training.

Initial experiment:

```text
one character
one LoRA
```

Do NOT immediately attempt all three characters and style simultaneously.

First prove the pipeline using one character.

Training configuration should expose important parameters:

```yaml
model:
training_resolution:
rank:
learning_rate:
steps:
batch_size:
gradient_accumulation:
mixed_precision:
seed:
trigger_token:
```

Save:

```text
LoRA weights
training config
training metadata
loss history
sample images
```

---

# 12. Experiment A — Baseline vs LoRA

Run the exact same benchmark prompts and seeds.

Compare:

```text
BASE MODEL

vs

BASE MODEL + CHARACTER LORA
```

Produce side-by-side grids.

This is the first important README result.

---

# 13. Identity Evaluation

Implement an embedding-based identity metric.

Concept:

```text
reference images
      ↓
image encoder
      ↓
reference embeddings
      ↓
character centroid


generated image
      ↓
image encoder
      ↓
generated embedding

      ↓

cosine similarity
```

Return:

```text
identity_similarity = 0..1
```

Important:

This metric is a proxy.

Do not describe it as objective proof that two images contain the same character.

Document its limitations.

---

# 14. Reference Crops

Background similarity can corrupt identity measurements.

Therefore evaluation should support pre-cropped character references.

Dataset format:

```text
references/
  dino/
    001.png
    002.png
```

V1 can use manually prepared crops.

Automatic detection/segmentation is a later experiment.

---

# 15. Style Evaluation

Create a reference set describing the target illustration style.

Calculate an embedding centroid.

Compare generated image embeddings against this style reference set.

Output:

```text
style_similarity
```

Again, document that this is a proxy metric.

---

# 16. Prompt Alignment

Measure whether improving character consistency causes the model to ignore the requested scene.

Example failure:

Prompt:

> Dino riding a bicycle.

Output:

> Dino standing normally because the LoRA has overfit training poses.

Therefore calculate a prompt/image similarity metric.

Output:

```text
prompt_alignment
```

---

# 17. Human Evaluation

Automated metrics are not enough.

Create a simple human evaluation format.

For every benchmark image:

```text
identity: 1–5
style: 1–5
prompt_adherence: 1–5
composition: 1–5

failure_tags:
  wrong_character
  identity_drift
  wrong_colors
  duplicate_character
  character_fusion
  missing_character
  wrong_pose
```

Store as JSON/CSV.

Later an HTML UI can make reviewing easier.

---

# 18. Experiment Result

Every experiment should generate:

```text
runs/<experiment>/

config.yaml
metadata.json
metrics.json

images/
comparison_grid.png
report.html
```

Example summary:

```text
Experiment: Dino LoRA v3

Identity
Baseline       0.61
LoRA           0.84

Style
Baseline       0.79
LoRA           0.86

Prompt alignment
Baseline       0.91
LoRA           0.87
```

This allows interesting conclusions such as:

> Identity improved significantly, but stronger LoRA weights reduced prompt adherence.

That trade-off is more valuable than simply showing nice images.

---

# 19. LoRA Strength Experiment

Automatically compare:

```text
0.4
0.6
0.8
1.0
1.2
```

Use identical:

```text
prompt
seed
model
```

Generate a chart:

```text
LoRA strength → identity score
LoRA strength → prompt alignment
```

Goal:

find the trade-off between character fidelity and model flexibility.

---

# 20. Training Data Experiment

Compare dataset sizes where practical.

Example:

```text
5 images
10 images
20 images
```

Question:

> How does dataset size affect identity consistency and overfitting?

This gives the repository a genuine ML experimentation component.

Do not promise a universal answer.

Report results for this dataset/model.

---

# 21. Reference Conditioning — Phase 2

After LoRA baseline works, add reference-image conditioning.

Support an experiment using an available Diffusers-compatible image adapter such as IP-Adapter where compatible with the selected checkpoint.

Compare:

```text
baseline
LoRA
reference conditioning
LoRA + reference conditioning
```

Use exactly the same benchmark.

Do not mix this work into the initial LoRA implementation.

---

# 22. Structural Control — Phase 3

Add a structural-control experiment using a supported ControlNet/pose/depth pipeline.

Purpose:

separate three different problems:

```text
Who is it?
→ character identity

How should it look?
→ style

Where/how should it appear?
→ composition / pose
```

Conceptual pipeline:

```text
Text prompt
      │
Character LoRA
      │
Reference image conditioning
      │
Structural control
      │
      ▼
Diffusion model
      │
      ▼
Generated image
```

Evaluate whether structural conditioning improves prompt/composition adherence without damaging identity.

---

# 23. Multiple Character Problem

After a single-character pipeline works, introduce:

```text
Dino + Tira
```

then:

```text
Dino + Tira + Dipo
```

Track failures:

```text
character fusion
attribute leakage
wrong character count
identity swapping
duplicate characters
missing characters
```

This should be treated as a separate research problem.

Do not assume that a LoRA working for one character will automatically solve multi-character scenes.

---

# 24. Character LoRA Strategy Experiment

Eventually compare:

```text
A. one joint LoRA containing all characters

vs

B. separate LoRA per character
```

For B:

```text
dino.safetensors
tira.safetensors
dipo.safetensors
```

If the backend permits multiple adapters, test combinations and adapter weights.

Document interference between concepts.

---

# 25. Style Separation Experiment

Compare:

```text
character + style learned together

vs

character LoRA + separate style LoRA
```

Question:

> Does separating character identity from artistic style improve generalization?

This can become one of the strongest experiments in the project.

---

# 26. Experiment CLI

Desired interface:

```bash
character-lab dataset validate datasets/dino

character-lab train \
  --config configs/training/dino.yaml

character-lab generate \
  --experiment configs/experiments/baseline.yaml

character-lab evaluate runs/baseline

character-lab compare \
  runs/baseline \
  runs/dino-lora-v1
```

Aim for reproducible CLI workflows rather than notebooks containing hidden state.

Notebooks may be added for exploration, but the real pipeline must live in Python modules.

---

# 27. Automated Report

Generate an HTML report.

Sections:

```text
Experiment configuration

Quantitative metrics

Baseline vs experiment

Identity examples

Best results

Worst results

Human evaluation

Failure categories

Conclusions
```

Include contact sheets.

This report should be viewable without running the model.

---

# 28. Tests

CI must NOT require a large GPU model download.

Unit-test:

```text
dataset validation
config parsing
manifest handling
experiment registry
metric calculations
report generation
seed handling
file structure
```

Mock the expensive model layer.

GPU integration tests should be separately marked.

Example:

```text
pytest -m gpu
```

Normal:

```text
pytest
```

must run on ordinary CI.

---

# 29. Model Weights

Never commit large pretrained models.

Never commit generated LoRA weights unless they are small enough and licensing permits it.

Prefer:

```text
release artifact
Hugging Face model repository
documented download
```

Keep licenses and model attribution explicit.

Training images must either be owned/licensed appropriately or replaced with a public demo dataset.

---

# 30. V1 Non-Goals

Do NOT initially build:

- book editor;
- story generator;
- text-generation agent;
- SaaS;
- user accounts;
- payments;
- mobile application;
- complex web frontend;
- video generation;
- model training from scratch.

The value of this repository is ML experimentation.

---

# 31. Milestones

## Milestone 0 — ML Project Foundation

Create:

- Python package;
- CLI;
- configuration system;
- tests;
- dataset abstraction;
- model backend interface.

Commit.

---

## Milestone 0.5 — Continuous Integration

Earlier versions of this brief assumed CI already existed. It did not. Nightly
sessions merged their own pull requests with no automated verification at all.

Required:

- a workflow running on every pull request and on pushes to `main`;
- install, build where applicable, and the full non-GPU test suite;
- no large model downloads — ordinary CI must stay fast and free;
- GPU-dependent tests marked so ordinary CI excludes them (section 28);
- the workflow registered as a required status check on `main`, so a red build
  blocks the merge instead of depending on the agent's own judgement.

A minimal workflow already exists at `.github/workflows/ci.yml`. Extending it
with linting, formatting and type checking belongs to this milestone, and
should happen when those tools are actually introduced — not before.

### If CI fails

This applies to every session, not only to this milestone.

1. **Do not merge.** A red build is never "clearly safe".
2. **Try to fix it in the same session** — but only if your own change caused
   it. Read the failing job's log, fix the cause, push to the same branch.
3. **If the failure is not yours** — `main` was already red when you started —
   stop feature work. Repairing `main` becomes the session's task, because
   every following session is blocked until it is green.
4. **If you cannot fix it**, leave the pull request open, do not open a second
   one, and write plainly in `docs/PROGRESS.md` and the daily work log what is
   broken and what you tried. The next session must begin by reading that and
   finishing the repair, not by starting something new.
5. **Never disable, skip, or weaken a test to make the build pass.** Deleting a
   failing assertion is not a fix; it hides the defect from every future
   session and from you.

---

## Milestone 1 — Dataset + Benchmark

Implement:

- manifest;
- validation;
- dataset stats;
- benchmark scene schema;
- reproducible seeds.

Commit.

---

## Milestone 2 — Baseline Inference

Implement model inference.

Generate benchmark from base model.

Save full experiment metadata.

Generate contact sheet.

Commit.

---

## Milestone 3 — LoRA Training

Implement/train one-character LoRA.

Save model and training metadata.

Commit code and small results, not large model files.

---

## Milestone 4 — Evaluation

Implement:

- identity metric;
- style metric;
- prompt alignment;
- human review format.

Compare:

```text
baseline vs LoRA
```

Generate HTML report.

Commit.

At this point the repository already qualifies as a meaningful V1.

---

## Milestone 5 — Hyperparameter Experiments

Add automated LoRA-strength comparison.

Optionally evaluate dataset size.

Generate plots and report.

Commit.

---

## Milestone 6 — Reference Conditioning

Add IP-Adapter or equivalent supported reference-image conditioning.

Compare against LoRA.

Commit.

---

## Milestone 7 — Structural Control

Add ControlNet/pose/depth experiment.

Commit.

---

## Milestone 8 — Multi-Character

Introduce two-character and three-character benchmark scenes.

Measure interference and failure modes.

Commit.

---

# 32. V1 Definition of Done

The project is V1-complete when the repository contains a reproducible experiment demonstrating:

```text
base diffusion model

vs

same base model + trained character LoRA
```

using:

- the same benchmark prompts;
- the same seeds;
- stored training configuration;
- quantitative metrics;
- side-by-side images;
- human evaluation;
- documented limitations.

README must contain actual results.

Not:

> LoRA should improve consistency.

But:

> On this benchmark, identity similarity increased from X to Y while prompt alignment changed from A to B.

Even a negative result is valid if the experiment is reproducible.

---

# 33. README Story

Suggested opening:

> I started this project after trying to generate a long illustrated children's story with recurring characters.
>
> Generating one good image was easy. Generating dozens of different scenes while preserving the same characters was not.
>
> Character Consistency Lab explores that problem as an ML engineering task: fine-tuning diffusion models, conditioning generation, building reproducible benchmarks, and measuring the trade-off between character identity and prompt adherence.

Then:

```text
Problem
Research questions
Dataset
Architecture
Training
Evaluation methodology
Experiments
Results
Failure analysis
Reproduction instructions
Limitations
Future work
```

The README should feel closer to a small engineering/research report than a product landing page.

---

# 34. Instructions to Coding Agent

Start implementing the project from Milestone 0.

Do not build a UI first.

Do not implement IP-Adapter or ControlNet before the baseline and LoRA experiment pipeline works.

Do not hide core logic inside notebooks.

Use typed configuration and reproducible experiment metadata.

Keep GPU-heavy components behind clean interfaces so normal tests can run without GPU access.

After each milestone:

1. run formatting;
2. run linting;
3. run unit tests;
4. verify CLI commands;
5. update README;
6. make a clean commit.

Do not fabricate experiment results.

If GPU training has not been executed, clearly mark results as pending.

The main goal is to create a technically credible ML repository showing:

- dataset engineering;
- diffusion inference;
- LoRA fine-tuning;
- experimental design;
- embeddings;
- evaluation;
- reproducibility;
- failure analysis.

Treat the illustrated-book scenario as the motivating benchmark, not as the product itself.