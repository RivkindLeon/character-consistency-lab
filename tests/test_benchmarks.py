import tempfile
import unittest
from pathlib import Path

from character_consistency_lab.benchmarks import BenchmarkSchemaError, load_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_repository_benchmark_is_fixed_and_covers_required_categories(self) -> None:
        benchmark = load_benchmark("benchmarks/scenes.yaml")

        self.assertEqual(benchmark.version, 1)
        self.assertEqual(len(benchmark.scenes), 20)
        self.assertEqual(len({scene.id for scene in benchmark.scenes}), 20)
        self.assertEqual(len({scene.seed for scene in benchmark.scenes}), 20)
        self.assertEqual(
            {scene.category for scene in benchmark.scenes},
            {
                "single_character_neutral", "unusual_pose", "close_up", "distant",
                "lighting_change", "background_change", "camera_angle",
                "two_characters", "three_characters", "complex_composition",
            },
        )

    def test_repeated_load_preserves_prompts_and_seeds(self) -> None:
        first = load_benchmark("benchmarks/scenes.yaml")
        second = load_benchmark("benchmarks/scenes.yaml")
        self.assertEqual(first, second)

    def test_rejects_invalid_seed(self) -> None:
        content = """
version: 1
scenes:
  - {id: same, category: neutral, characters: [dino], prompt: first, seed: 1}
  - {id: same, category: neutral, characters: [dino], prompt: second, seed: -1}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenes.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkSchemaError, "scene.seed"):
                load_benchmark(path)

    def test_rejects_duplicate_ids(self) -> None:
        content = """
version: 1
scenes:
  - {id: same, category: neutral, characters: [dino], prompt: first, seed: 1}
  - {id: same, category: neutral, characters: [dino], prompt: second, seed: 2}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenes.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkSchemaError, "scene IDs"):
                load_benchmark(path)

    def test_rejects_unknown_fields(self) -> None:
        content = """
version: 1
scenes:
  - id: dino
    category: neutral
    characters: [dino]
    prompt: chr_dino standing
    seed: 1
    typo: true
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "scenes.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkSchemaError, "typo"):
                load_benchmark(path)


if __name__ == "__main__":
    unittest.main()
