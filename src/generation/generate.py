from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import torch

from .pipeline import SDConfig, SDMPSPipeline


Resolution = Tuple[int, int]


ALLOWED_RESOLUTIONS: List[Resolution] = [
    (512, 512),
    (512, 768),
    (768, 512),
]


class ResolutionError(ValueError):
    pass


def validate_resolution(height: int, width: int) -> None:
    if (height, width) not in ALLOWED_RESOLUTIONS:
        raise ResolutionError(
            f"Unsupported resolution {height}x{width}. "
            f"Allowed: {', '.join(f'{h}x{w}' for h, w in ALLOWED_RESOLUTIONS)}"
        )


def _build_pipeline(base_config: SDConfig) -> SDMPSPipeline:
    # Guardrail: avoid float64 issues on MPS by ensuring default dtypes are safe.
    torch.set_default_dtype(torch.float32)

    pipe_wrapper = SDMPSPipeline(config=base_config)
    return pipe_wrapper


def generate_batch(
    prompt: str,
    *,
    negative_prompt: Optional[str] = None,
    base_seed: Optional[int] = None,
    num_images: int = 1,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    model_id: str = "runwayml/stable-diffusion-v1-5",
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Generate one or more images with deterministic seeding and full metadata.

    - Same (prompt, negative prompt, seed, steps, guidance, model, resolution) should
      yield the same outputs as closely as possible.
    - Batch generation uses sequential seeds: base_seed, base_seed+1, ...
    """

    if num_images < 1:
        raise ValueError("num_images must be >= 1")

    validate_resolution(height, width)

    config = SDConfig(
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=base_seed,
    )

    wrapper = _build_pipeline(config)
    pipe = wrapper.pipe

    images: List[Any] = []
    metadata_list: List[Dict[str, Any]] = []

    for i in range(num_images):
        # Deterministic seeding per variation.
        effective_seed = (base_seed or 0) + i
        generator = torch.Generator(device=wrapper.device).manual_seed(effective_seed)

        t0 = perf_counter()
        result = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        duration = perf_counter() - t0

        image = result.images[0]
        images.append(image)

        metadata: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "seed": effective_seed,
            "base_seed": base_seed,
            "steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "model_id": model_id,
            "duration_sec": duration,
            "device": str(wrapper.device),
            "config": asdict(config),
            "variation_index": i,
            "num_images": num_images,
        }
        metadata_list.append(metadata)

    return images, metadata_list


__all__ = ["generate_batch", "ResolutionError", "validate_resolution", "ALLOWED_RESOLUTIONS"]

