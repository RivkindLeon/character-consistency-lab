import builtins
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from character_consistency_lab.config import ConfigurationError
from character_consistency_lab.models import (
    GenerationRequest,
    check_backend_runtime,
    create_backend,
    load_backend_config,
)


class DiffusersBackendConfigTests(unittest.TestCase):
    def test_loads_flux_configuration(self) -> None:
        config = load_backend_config("configs/models/flux2-klein-base-4b.yaml")
        self.assertEqual(config.backend, "flux")
        self.assertEqual(config.dtype, "bfloat16")

    def test_rejects_unknown_fields_and_backends(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text("backend: unknown\nmodel_id: model\nextra: true\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_backend_config(path)

    def test_dry_run_does_not_import_heavyweight_libraries(self) -> None:
        config = load_backend_config("configs/models/flux2-klein-base-4b.yaml")
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name in {"torch", "diffusers", "transformers"}:
                raise AssertionError(f"unexpected heavyweight import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            backend = create_backend(config, dry_run=True)
            backend.load()
        self.assertEqual(backend.name, "flux-diffusers-dry-run")

    def test_real_backend_contract_with_mocked_expensive_layer(self) -> None:
        class FakeImage:
            def save(self, path):
                Path(path).write_bytes(b"fake-image")

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, model_id, **kwargs):
                instance = cls()
                instance.load_args = (model_id, kwargs)
                return instance

            def to(self, device):
                self.device = device
                return self

            def __call__(self, **kwargs):
                self.generation_args = kwargs
                return types.SimpleNamespace(images=[FakeImage()])

        class FakeGenerator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        fake_diffusers = types.SimpleNamespace(DiffusionPipeline=FakePipeline)
        fake_torch = types.SimpleNamespace(bfloat16="bf16", Generator=FakeGenerator)
        config = load_backend_config("configs/models/flux2-klein-base-4b.yaml")

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules, {"diffusers": fake_diffusers, "torch": fake_torch}
        ):
            output = Path(directory) / "scene.png"
            backend = create_backend(config, dry_run=False)
            with backend:
                result = backend.generate(
                    GenerationRequest(prompt="chr_dino running", seed=41, output_path=output)
                )
            self.assertEqual(output.read_bytes(), b"fake-image")
            self.assertEqual(result.image_path, output)
            self.assertEqual(result.backend, "flux-diffusers")

    def test_runtime_check_verifies_cuda_without_loading_model(self) -> None:
        cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True)
        modules = {
            "torch": types.SimpleNamespace(__version__="2.8", cuda=cuda),
            "diffusers": types.SimpleNamespace(__version__="0.35"),
            "transformers": types.SimpleNamespace(),
            "accelerate": types.SimpleNamespace(),
            "safetensors": types.SimpleNamespace(),
        }
        config = load_backend_config("configs/models/flux2-klein-base-4b.yaml")
        with patch(
            "character_consistency_lab.models.diffusers.importlib.import_module",
            side_effect=lambda name: modules[name],
        ):
            result = check_backend_runtime(config)
        self.assertEqual(result["device"], "cuda")
        self.assertEqual(result["torch"], "2.8")

    def test_runtime_check_rejects_host_without_cuda(self) -> None:
        cuda = types.SimpleNamespace(is_available=lambda: False, is_bf16_supported=lambda: False)
        modules = {
            name: types.SimpleNamespace(cuda=cuda) if name == "torch" else types.SimpleNamespace()
            for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors")
        }
        config = load_backend_config("configs/models/flux2-klein-base-4b.yaml")
        with patch(
            "character_consistency_lab.models.diffusers.importlib.import_module",
            side_effect=lambda name: modules[name],
        ), self.assertRaisesRegex(ConfigurationError, "CUDA is not available"):
            check_backend_runtime(config)


if __name__ == "__main__":
    unittest.main()
