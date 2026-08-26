import json
import tempfile
import unittest
from pathlib import Path

from character_consistency_lab.manifest import generate_manifest, load_spec, manifest_to_json


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
    def test_generate_manifest_expands_variant_matrix(self) -> None:
        manifest = generate_manifest(SPEC)

        self.assertEqual(manifest["sample_count"], 4)
        self.assertEqual(len(manifest["samples"]), 4)
        self.assertIn("character: Ari", manifest["samples"][0]["prompt"])
        self.assertIn("same character", manifest["samples"][0]["prompt"])

    def test_seed_is_stable_for_same_spec(self) -> None:
        first = generate_manifest(SPEC)
        second = generate_manifest(SPEC)

        self.assertEqual(first["samples"][0]["seed"], second["samples"][0]["seed"])
        self.assertEqual(first["samples"][0]["sample_id"], second["samples"][0]["sample_id"])

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


if __name__ == "__main__":
    unittest.main()
