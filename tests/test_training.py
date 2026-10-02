from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image

from character_consistency_lab.config import ConfigurationError
from character_consistency_lab.training import (
    check_training_runtime,
    create_training_plan,
    load_training_config,
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


if __name__ == "__main__":
    unittest.main()
