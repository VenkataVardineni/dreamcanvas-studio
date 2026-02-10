from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline


@dataclass
class SDConfig:
    """Configuration for Stable Diffusion generation.

    This is kept small and explicit for reproducibility and easy logging.
    """

    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: Optional[int] = None


class SDMPSPipeline:
    """Thin wrapper around diffusers StableDiffusionPipeline with Apple Silicon (MPS) support.

    - Prefers MPS device on Apple Silicon, falls back to CPU.
    - Runs a one-time warmup inference on first use to stabilize performance.
    - Avoids unsupported float64 usage on MPS.
    """

    def __init__(self, config: Optional[SDConfig] = None) -> None:
        self.config = config or SDConfig()

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self._pipe: Optional[StableDiffusionPipeline] = None
        self._warmed_up: bool = False

    def _load_pipeline(self) -> StableDiffusionPipeline:
        if self._pipe is not None:
            return self._pipe

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device.type == "mps" else torch.float32,
        )

        # Safety: ensure no float64 tensors on MPS which can cause errors.
        pipe = pipe.to(self.device)

        # Enable memory-efficient attention if available
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing("max")

        self._pipe = pipe
        return pipe

    def _warmup(self) -> None:
        if self._warmed_up:
            return

        pipe = self._load_pipeline()

        # Run a single-step warmup with a trivial prompt.
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(self.config.seed)

        _ = pipe(
            "warmup image of a simple object",
            num_inference_steps=1,
            guidance_scale=1.0,
            height=self.config.height,
            width=self.config.width,
            generator=generator,
        )

        self._warmed_up = True

    @property
    def pipe(self) -> StableDiffusionPipeline:
        """Access the underlying diffusers pipeline, ensuring it is loaded and warmed up."""

        self._warmup()
        assert self._pipe is not None
        return self._pipe


__all__ = ["SDConfig", "SDMPSPipeline"]

