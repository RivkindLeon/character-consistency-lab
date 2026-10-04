from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image

from character_consistency_lab.config import ConfigurationError
from character_consistency_lab.training import (
    TrainingResult,
    TrainingStep,
    build_diffusers_training_command,
    check_training_runtime,
    create_training_plan,
    format_training_command,
    load_training_config,
    prepare_diffusers_dataset,
    run_training,
)


VALID_CONFIG = """\
model: example/model
dataset: datasets/dino
output_dir: runs/dino-lora
training_resolution: 1024
rank: 16
learning_rate: 0.0001
steps: 1000
batch_size: 2
gradient_accumulation: 4
mixed_precision: bf16
seed: 42137
trigger_token: chr_dino
"""


class TrainingConfigTests(unittest.TestCase):
    def _write_dataset(self, root: Path) -> None:
        dataset = root / "datasets" / "dino"
        (dataset / "images").mkdir(parents=True)
        Image.new("RGB", (64, 64)).save(dataset / "images" / "dino.png")
        (dataset / "characters.yaml").write_text(
            "characters:\n  - id: dino\n    trigger: chr_dino\n", encoding="utf-8"
        )
        (dataset / "manifest.jsonl").write_text(
            '{"image":"images/dino.png","character":"dino","caption":"chr_dino portrait","split":"train"}\n',
            encoding="utf-8",
        )

    def test_loads_complete_config_and_builds_cpu_safe_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "configs" / "training" / "dino.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(VALID_CONFIG, encoding="utf-8")

            config = load_training_config(config_path)
            plan = create_training_plan(config, config_path)

            self.assertEqual(config.rank, 16)
            self.assertEqual(plan.effective_batch_size, 8)
            self.assertEqual(plan.dataset, root / "datasets" / "dino")
            self.assertEqual(plan.output_dir, root / "runs" / "dino-lora")

    def test_rejects_unknown_and_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "bad.yaml"
            config_path.write_text(
                VALID_CONFIG.replace("rank: 16", "rank: 0") + "unknown: value\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_training_config(config_path)

    def test_rejects_missing_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "bad.yaml"
            config_path.write_text("model: example/model\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_training_config(config_path)

    def test_runtime_preflight_checks_dependencies_cuda_and_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root)
            config_path = root / "configs" / "training" / "dino.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(VALID_CONFIG, encoding="utf-8")
            config = load_training_config(config_path)
            plan = create_training_plan(config, config_path)
            cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)

            def load_module(name: str) -> SimpleNamespace:
                if name == "torch":
                    return SimpleNamespace(__version__="2.test", cuda=cuda)
                return SimpleNamespace(__version__=f"{name}.test")

            with patch(
                "character_consistency_lab.training.lora.importlib.import_module",
                side_effect=load_module,
            ):
                result = check_training_runtime(plan, config)

            self.assertEqual(result["dataset_images"], 1)
            self.assertEqual(result["peft"], "peft.test")

    def test_runtime_preflight_reports_missing_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root)
            config_path = root / "configs" / "training" / "dino.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(VALID_CONFIG, encoding="utf-8")
            config = load_training_config(config_path)
            plan = create_training_plan(config, config_path)

            with patch(
                "character_consistency_lab.training.lora.importlib.import_module",
                side_effect=ImportError,
            ), self.assertRaisesRegex(ConfigurationError, "install.*training"):
                check_training_runtime(plan, config)

    def test_prepares_train_only_imagefolder_with_manifest_captions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root)
            dataset = root / "datasets" / "dino"
            Image.new("RGB", (64, 64), color="blue").save(
                dataset / "images" / "reference.png"
            )
            with (dataset / "manifest.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    '{"image":"images/reference.png","character":"dino",'
                    '"caption":"reference only","split":"reference"}\n'
                )
            config_path = root / "configs" / "training" / "dino.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(VALID_CONFIG, encoding="utf-8")
            config = load_training_config(config_path)
            plan = create_training_plan(config, config_path)

            prepared = prepare_diffusers_dataset(plan)

            self.assertEqual(
                sorted(path.name for path in prepared.iterdir()),
                ["0001.png", "metadata.jsonl"],
            )
            self.assertEqual(
                (prepared / "metadata.jsonl").read_text(encoding="utf-8"),
                '{"file_name": "0001.png", "text": "chr_dino portrait"}\n',
            )
            with self.assertRaisesRegex(ConfigurationError, "already exists"):
                prepare_diffusers_dataset(plan)

    def test_builds_shell_safe_official_diffusers_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "configs" / "training" / "dino.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(VALID_CONFIG, encoding="utf-8")
            config = load_training_config(config_path)
            plan = create_training_plan(config, config_path)

            command = build_diffusers_training_command(
                plan, config, root / "scripts with spaces" / "trainer.py"
            )

            self.assertEqual(command[:2], ("accelerate", "launch"))
            self.assertIn("--caption_column", command)
            self.assertEqual(command[command.index("--rank") + 1], "16")
            self.assertEqual(command[command.index("--max_train_steps") + 1], "1000")
            self.assertIn("'", format_training_command(command))

    def test_runner_persists_complete_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root)
            config_path = root / "configs" / "training" / "dino.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                VALID_CONFIG.replace("steps: 1000", "steps: 2"), encoding="utf-8"
            )
            config = load_training_config(config_path)
            plan = create_training_plan(config, config_path)

            class FakeBackend:
                def train(self, plan, config, on_step):
                    history = (TrainingStep(1, 0.75), TrainingStep(2, 0.5))
                    for item in history:
                        on_step(item)
                    weights = plan.output_dir / "dino.safetensors"
                    weights.write_bytes(b"fake weights")
                    sample = plan.output_dir / "sample.png"
                    Image.new("RGB", (16, 16)).save(sample)
                    return TrainingResult(
                        weights_path=weights,
                        loss_history=history,
                        sample_paths=(sample,),
                        backend_metadata={"name": "fake"},
                    )

            metadata_path = run_training(plan, config, FakeBackend())

            import json

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            losses = json.loads(
                (plan.output_dir / "loss_history.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["dataset_images"], 1)
            self.assertEqual(metadata["backend"], {"name": "fake"})
            self.assertEqual(losses, [{"step": 1, "loss": 0.75}, {"step": 2, "loss": 0.5}])
            self.assertTrue((plan.output_dir / "config.yaml").is_file())

    def test_runner_checkpoints_failure_without_claiming_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_dataset(root)
            config_path = root / "configs" / "training" / "dino.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(
                VALID_CONFIG.replace("steps: 1000", "steps: 2"), encoding="utf-8"
            )
            config = load_training_config(config_path)
            plan = create_training_plan(config, config_path)

            class FailingBackend:
                def train(self, plan, config, on_step):
                    on_step(TrainingStep(1, 0.75))
                    raise RuntimeError("GPU disconnected")

            with self.assertRaisesRegex(RuntimeError, "GPU disconnected"):
                run_training(plan, config, FailingBackend())

            import json

            metadata = json.loads(
                (plan.output_dir / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["status"], "failed")
            self.assertIsNone(metadata["weights"])


if __name__ == "__main__":
    unittest.main()
