import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from character_consistency_lab.experiments import load_experiment_config, run_experiment
from character_consistency_lab.models import GenerationResult


class _FixtureBackend:
    name = "fixture"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def generate(self, request):
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "navy").save(request.output_path)
        return GenerationResult(
            backend=self.name,
            model_id="fixture/model",
            model_revision="test",
            request=request,
            image_path=request.output_path,
        )


class ExperimentRunnerTests(unittest.TestCase):
    def test_baseline_config_has_no_adapter(self) -> None:
        config = load_experiment_config("configs/experiments/baseline.yaml")
        self.assertEqual(config.name, "baseline")
        self.assertEqual(config.generation.adapter_configuration, {})

    def test_dry_run_records_every_scene_and_full_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiment.yaml"
            config.write_text(
                "\n".join(
                    [
                        "name: test-baseline",
                        "benchmark: benchmarks/scenes.yaml",
                        "model: configs/models/flux2-klein-base-4b.yaml",
                        f"output_dir: {root / 'run'}",
                        "generation:",
                        "  width: 640",
                        "  height: 768",
                        "  steps: 20",
                        "  guidance: 4.0",
                        "  negative_prompt: blurry",
                        "  adapter_configuration: {}",
                    ]
                ),
                encoding="utf-8",
            )
            metadata_path = run_experiment(config, dry_run=True)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["generation_count"], 20)
            self.assertTrue(metadata["dry_run"])
            self.assertTrue(metadata["timestamp"].endswith("+00:00"))
            self.assertIn("git_commit_sha", metadata)
            self.assertIsNone(metadata["contact_sheet"])
            first = metadata["generations"][0]
            required = {
                "model", "model_revision", "lora", "lora_weight", "prompt",
                "negative_prompt", "seed", "width", "height", "steps", "guidance",
                "adapter_configuration", "planned_image", "dry_run",
            }
            self.assertTrue(required.issubset(first))
            self.assertIsNone(first["image"])
            self.assertFalse((root / "run" / "images").exists())
            self.assertTrue((root / "run" / "config.yaml").exists())
            self.assertFalse((root / "run" / "comparison_grid.png").exists())

    def test_completed_run_creates_and_records_contact_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiment.yaml"
            config.write_text(
                "\n".join(
                    [
                        "name: fixture-run",
                        "benchmark: benchmarks/scenes.yaml",
                        "model: configs/models/flux2-klein-base-4b.yaml",
                        f"output_dir: {root / 'run'}",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "character_consistency_lab.experiments.runner.create_backend",
                return_value=_FixtureBackend(),
            ):
                metadata_path = run_experiment(config, dry_run=False)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            grid = root / "run" / "comparison_grid.png"
            self.assertEqual(metadata["contact_sheet"], str(grid))
            self.assertTrue(grid.is_file())
            self.assertEqual(len(list((root / "run" / "images").glob("*.png"))), 20)


if __name__ == "__main__":
    unittest.main()
