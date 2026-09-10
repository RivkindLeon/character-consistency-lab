import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from character_consistency_lab.config import ConfigurationError
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
            model_id="black-forest-labs/FLUX.2-klein-4B",
            model_revision=None,
            request=request,
            image_path=request.output_path,
        )


class _FailingFixtureBackend(_FixtureBackend):
    def __init__(self, fail_after):
        self.calls = 0
        self.fail_after = fail_after

    def generate(self, request):
        if self.calls == self.fail_after:
            raise RuntimeError("simulated remote interruption")
        self.calls += 1
        return super().generate(request)


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
            self.assertEqual(metadata["status"], "completed")
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

    def test_real_run_checkpoints_and_resumes_after_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            config = root / "experiment.yaml"
            config.write_text(
                "\n".join(
                    [
                        "name: resumable-run",
                        "benchmark: benchmarks/scenes.yaml",
                        "model: configs/models/flux2-klein-base-4b.yaml",
                        f"output_dir: {run_dir}",
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "character_consistency_lab.experiments.runner.create_backend",
                return_value=_FailingFixtureBackend(fail_after=3),
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated remote interruption"):
                    run_experiment(config, dry_run=False)

            checkpoint = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["status"], "running")
            self.assertEqual(checkpoint["generation_count"], 3)
            self.assertTrue((run_dir / "config.yaml").is_file())

            resumed_backend = _FixtureBackend()
            with patch(
                "character_consistency_lab.experiments.runner.create_backend",
                return_value=resumed_backend,
            ):
                metadata_path = run_experiment(config, dry_run=False, resume=True)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["generation_count"], 20)
            self.assertEqual(len(metadata["generations"]), 20)
            self.assertTrue((run_dir / "comparison_grid.png").is_file())

    def test_resume_rejects_changed_generation_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "experiment.yaml"
            config.write_text(
                "\n".join(
                    [
                        "name: changed-run",
                        "benchmark: benchmarks/scenes.yaml",
                        "model: configs/models/flux2-klein-base-4b.yaml",
                        f"output_dir: {root / 'run'}",
                    ]
                ),
                encoding="utf-8",
            )
            with patch(
                "character_consistency_lab.experiments.runner.create_backend",
                return_value=_FailingFixtureBackend(fail_after=1),
            ):
                with self.assertRaises(RuntimeError):
                    run_experiment(config, dry_run=False)

            config.write_text(config.read_text(encoding="utf-8") + "\ngeneration:\n  steps: 10\n")
            with patch(
                "character_consistency_lab.experiments.runner.create_backend",
                return_value=_FixtureBackend(),
            ):
                with self.assertRaisesRegex(ConfigurationError, "configuration changed"):
                    run_experiment(config, dry_run=False, resume=True)


if __name__ == "__main__":
    unittest.main()
