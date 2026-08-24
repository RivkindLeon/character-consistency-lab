# Character Consistency Lab

Utilities and experiments for preserving character identity consistency across AI-generated images.

## Included today

- `ccl-manifest`: expands a TOML experiment spec into a reproducible JSON prompt manifest
- deterministic seed derivation per sample for repeatable runs
- prompt-locked scene variation grids for character consistency sweeps
- render-parameter sweeps for guidance scale, LoRA strength, steps, and canvas size

## Why this helps

Character-consistency work usually needs two things early:

1. a stable identity description that stays locked across generations
2. a controlled variation grid for pose, expression, lighting, and background

In practice, you also need to test generation settings that can preserve or destroy identity. This repo now includes a small generator that turns both prompt variation and render sweeps into a concrete manifest you can feed into Diffusers or other experiment runners.

## Quick start

```bash
python -m venv venv
./venv/bin/pip install -e .
./venv/bin/ccl-manifest \
  --spec examples/mira_consistency.toml \
  --output out/mira_consistency.json
```

Example output fields:

- `sample_id`
- `seed`
- `prompt`
- `negative_prompt`
- `tags`
- `render_settings`

## Spec format

```toml
[experiment]
name = "mira-consistency-v1"
base_prompt = "highly detailed cinematic illustration"
negative_prompt = "lowres, blurry, duplicated face, extra limbs"
base_seed = 4242

[character]
name = "Mira Vale"
identity = ["short silver bob haircut", "amber eyes"]

[consistency]
always = ["same adult woman", "same face shape", "same hairstyle"]

[variants]
shots = ["portrait close-up", "full body"]
expressions = ["calm confidence", "determined focus"]
actions = ["standing still"]
backgrounds = ["neon rainy alley", "sunlit train platform"]
outfits = ["red trench coat"]
lighting = ["soft rim light"]

[render]
width = 768
height = 1024
num_inference_steps = 32

[sweeps]
guidance_scales = [6.0, 7.5]
lora_scales = [0.65, 0.85]
```

The generator creates the Cartesian product of the provided prompt variants and render sweeps, while keeping identity + consistency locks in every prompt.

## Run tests

```bash
PYTHONPATH=src ./venv/bin/python -m unittest discover -s tests
```
