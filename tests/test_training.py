from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from character_consistency_lab.config import ConfigurationError
from character_consistency_lab.training import create_training_plan, load_training_config


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


if __name__ == "__main__":
    unittest.main()
