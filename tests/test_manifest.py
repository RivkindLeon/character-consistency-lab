import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from character_consistency_lab.manifest import (
    SpecValidationError,
    generate_manifest,
    load_spec,
    manifest_to_json,
    validate_spec,
)


SPEC = {
    "experiment": {
        "name": "hero-v1",
        "base_prompt": "stylized illustration",
        "negative_prompt": "blurry",
        "base_seed": 100,
    },
    "character": {
        "name": "Ari",
        "identity": ["green eyes", "freckles"],
    },
    "consistency": {
        "always": ["same character", "same face"],
    },
    "variants": {
        "shots": ["portrait", "full body"],
        "expressions": ["neutral", "smiling"],
        "actions": ["standing"],
    },
}


class ManifestTests(unittest.TestCase):
    def test_validate_spec_accepts_valid_spec(self) -> None:
        validate_spec(SPEC)

    def test_validate_spec_requires_experiment_name(self) -> None:
        spec = {
            **SPEC,
            "experiment": {
                **SPEC["experiment"],
                "name": " ",
            },
        }

        with self.assertRaisesRegex(SpecValidationError, "experiment.name"):
            validate_spec(spec)

    def test_validate_spec_rejects_empty_sweep_values(self) -> None:
        spec = {
            **SPEC,
            "sweeps": {
                "guidance_scales": [],
            },
        }

        with self.assertRaisesRegex(SpecValidationError, "sweeps.guidance_scales"):
            validate_spec(spec)

    def test_validate_spec_rejects_non_positive_render_dimensions(self) -> None:
        spec = {
            **SPEC,
            "render": {
                "width": 0,
            },
        }

        with self.assertRaisesRegex(SpecValidationError, "render.width"):
            validate_spec(spec)

    def test_generate_manifest_expands_variant_matrix(self) -> None:
        manifest = generate_manifest(SPEC)

        self.assertEqual(manifest["sample_count"], 4)
        self.assertEqual(len(manifest["samples"]), 4)
        self.assertEqual(len(manifest["comparison_groups"]), 4)
        self.assertIn("character: Ari", manifest["samples"][0]["prompt"])
        self.assertIn("same character", manifest["samples"][0]["prompt"])
        self.assertEqual(
            manifest["samples"][0]["comparison_group_id"],
            manifest["comparison_groups"][0]["comparison_group_id"],
        )

    def test_seed_is_stable_for_same_spec(self) -> None:
        first = generate_manifest(SPEC)
        second = generate_manifest(SPEC)

        self.assertEqual(first["samples"][0]["seed"], second["samples"][0]["seed"])
        self.assertEqual(first["samples"][0]["sample_id"], second["samples"][0]["sample_id"])

    def test_render_variants_share_a_paired_seed_within_each_scene(self) -> None:
        spec = {
            **SPEC,
            "sweeps": {
                "guidance_scales": [5.0, 7.5],
                "lora_scales": [0.6, 0.9],
            },
        }

        manifest = generate_manifest(spec)
        seeds_by_group: dict[str, set[int]] = {}
        for sample in manifest["samples"]:
            seeds_by_group.setdefault(sample["comparison_group_id"], set()).add(sample["seed"])

        self.assertTrue(all(len(seeds) == 1 for seeds in seeds_by_group.values()))
        self.assertEqual(len({next(iter(seeds)) for seeds in seeds_by_group.values()}), 4)

    def test_render_sweeps_expand_samples_and_emit_render_settings(self) -> None:
        spec = {
            **SPEC,
            "render": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "lora_adapter": "loras/ari_v1.safetensors",
                "width": 768,
                "height": 1024,
                "guidance_scale": 6.5,
            },
            "sweeps": {
                "model_ids": [
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    "RunDiffusion/Juggernaut-XL-v9",
                ],
                "lora_adapters": ["loras/ari_v1.safetensors"],
                "guidance_scales": [6.5, 8.0],
                "num_inference_steps": [28, 36],
                "lora_scales": [0.7],
            },
        }

        manifest = generate_manifest(spec)

        self.assertEqual(manifest["sample_count"], 32)
        self.assertEqual(len(manifest["comparison_groups"]), 4)
        self.assertEqual(
            manifest["render"]["model_id"],
            "stabilityai/stable-diffusion-xl-base-1.0",
        )
        self.assertEqual(
            manifest["samples"][0]["render_settings"]["lora_adapter"],
            "loras/ari_v1.safetensors",
        )
        self.assertEqual(manifest["samples"][0]["render_settings"]["height"], 1024)
        self.assertIn(manifest["samples"][0]["render_settings"]["guidance_scale"], {6.5, 8.0})
        self.assertIn("model-stabilityai/stable-diffusion-xl-base-1p0", manifest["samples"][0]["sample_id"])
        self.assertIn("adapter-loras/ari_v1psafetensors", manifest["samples"][0]["sample_id"])
        self.assertIn("steps-28", manifest["samples"][0]["sample_id"])
        self.assertEqual(
            manifest["samples"][0]["comparison_group_id"],
            manifest["samples"][1]["comparison_group_id"],
        )
        self.assertEqual(len(manifest["comparison_groups"][0]["sample_ids"]), 8)

    def test_render_defaults_are_copied_without_sweeps(self) -> None:
        spec = {
            **SPEC,
            "render": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "lora_adapter": "loras/ari_v1.safetensors",
                "width": 640,
                "height": 960,
                "num_inference_steps": 30,
            },
        }

        manifest = generate_manifest(spec)

        self.assertEqual(manifest["sample_count"], 4)
        self.assertEqual(
            manifest["samples"][0]["render_settings"],
            {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "lora_adapter": "loras/ari_v1.safetensors",
                "width": 640,
                "height": 960,
                "num_inference_steps": 30,
            },
        )
        self.assertEqual(len(manifest["comparison_groups"]), 4)
        self.assertEqual(len(manifest["comparison_groups"][0]["sample_ids"]), 1)

    def test_load_spec_from_toml(self) -> None:
        content = """
[experiment]
name = "hero-v1"
base_prompt = "stylized illustration"

[variants]
shots = ["portrait"]
""".strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.toml"
            path.write_text(content, encoding="utf-8")
            loaded = load_spec(path)

        self.assertEqual(loaded["experiment"]["name"], "hero-v1")
        self.assertEqual(loaded["variants"]["shots"], ["portrait"])

    def test_manifest_to_json_is_valid_json(self) -> None:
        manifest = generate_manifest(SPEC)
        payload = manifest_to_json(manifest)

        parsed = json.loads(payload)
        self.assertEqual(parsed["sample_count"], 4)
        self.assertIn("comparison_group_id", parsed["samples"][0])

    def test_validate_spec_cli_reports_success(self) -> None:
        content = """
[experiment]
name = "hero-v1"
base_prompt = "stylized illustration"

[variants]
shots = ["portrait"]
""".strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.toml"
            path.write_text(content, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "character_consistency_lab.cli",
                    "validate-spec",
                    "--spec",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Spec is valid", result.stdout)

    def test_validate_spec_cli_reports_failure(self) -> None:
        content = """
[experiment]
name = ""
base_prompt = "stylized illustration"
""".strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.toml"
            path.write_text(content, encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "character_consistency_lab.cli",
                    "validate-spec",
                    "--spec",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Spec validation failed", result.stderr)

    def test_validate_spec_cli_reports_malformed_toml_without_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "spec.toml"
            path.write_text('[experiment\nname = "broken"', encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "character_consistency_lab.cli",
                    "validate-spec",
                    "--spec",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Spec validation failed", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
