"""Backend-neutral contracts for diffusion model inference.

Concrete backends may import heavyweight ML libraries, but this module must stay
lightweight so dataset preparation, configuration validation, and dry runs work
on machines without a GPU or model cache.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class GenerationRequest:
    """A reproducible, backend-independent image generation request."""

    prompt: str
    seed: int
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 3.5
    negative_prompt: str | None = None
    output_path: Path | None = None
    adapter_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name in ("width", "height", "steps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.guidance < 0:
            raise ValueError("guidance must be non-negative")


@dataclass(frozen=True)
class GenerationResult:
    """The artifact and metadata returned by a generation backend."""

    backend: str
    model_id: str
    model_revision: str | None
    request: GenerationRequest
    image_path: Path | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dry_run: bool = False


class ModelBackend(ABC):
    """Interface implemented by FLUX, SDXL, and test/dry-run backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable backend identifier used in experiment metadata."""

    @abstractmethod
    def load(self) -> None:
        """Load model resources, or validate them for a non-executing backend."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one image from a validated request."""

    @abstractmethod
    def unload(self) -> None:
        """Release backend resources. Calling this repeatedly must be safe."""

    def __enter__(self) -> ModelBackend:
        self.load()
        return self

    def __exit__(self, *_: object) -> None:
        self.unload()


class DryRunBackend(ModelBackend):
    """CPU-safe backend that records intent without loading or writing a model."""

    def __init__(
        self,
        model_id: str,
        model_revision: str | None = None,
        *,
        backend_name: str = "dry-run",
    ) -> None:
        if not model_id.strip():
            raise ValueError("model_id must not be empty")
        self.model_id = model_id
        self.model_revision = model_revision
        self.backend_name = backend_name
        self.is_loaded = False

    @property
    def name(self) -> str:
        return self.backend_name

    def load(self) -> None:
        self.is_loaded = True

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("backend must be loaded before generation")
        return GenerationResult(
            backend=self.name,
            model_id=self.model_id,
            model_revision=self.model_revision,
            request=request,
            image_path=None,
            metadata={
                "planned_output_path": (
                    str(request.output_path) if request.output_path is not None else None
                )
            },
            dry_run=True,
        )

    def unload(self) -> None:
        self.is_loaded = False
