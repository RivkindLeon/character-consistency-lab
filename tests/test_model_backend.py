import unittest
from pathlib import Path

from character_consistency_lab.models import DryRunBackend, GenerationRequest


class GenerationRequestTests(unittest.TestCase):
    def test_rejects_invalid_generation_values(self) -> None:
        invalid = (
            {"prompt": " ", "seed": 1},
            {"prompt": "portrait", "seed": -1},
            {"prompt": "portrait", "seed": 1, "width": 0},
            {"prompt": "portrait", "seed": 1, "guidance": -0.1},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                GenerationRequest(**values)


class DryRunBackendTests(unittest.TestCase):
    def test_requires_load_before_generate(self) -> None:
        backend = DryRunBackend("black-forest-labs/FLUX.2-klein-base-4B")
        request = GenerationRequest(prompt="chr_dino in a meadow", seed=42)

        with self.assertRaisesRegex(RuntimeError, "must be loaded"):
            backend.generate(request)

    def test_context_manager_returns_metadata_without_creating_image(self) -> None:
        output = Path("runs/dry-run/images/dino.png")
        request = GenerationRequest(
            prompt="chr_dino in a meadow",
            negative_prompt="blurry",
            seed=42,
            output_path=output,
            adapter_config={"lora": "dino.safetensors", "scale": 0.8},
        )
        backend = DryRunBackend(
            "black-forest-labs/FLUX.2-klein-base-4B",
            model_revision="abc123",
        )

        with backend:
            result = backend.generate(request)

        self.assertTrue(result.dry_run)
        self.assertEqual(result.backend, "dry-run")
        self.assertEqual(result.model_revision, "abc123")
        self.assertEqual(result.request.seed, 42)
        self.assertEqual(result.metadata["planned_output_path"], str(output))
        self.assertIsNone(result.image_path)
        self.assertFalse(output.exists())
        self.assertFalse(backend.is_loaded)


if __name__ == "__main__":
    unittest.main()
