from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class StylePreset:
    id: str
    name: str
    style_prefix: str
    negative_prompt: str


_PRESETS_CACHE: Optional[Dict[str, StylePreset]] = None


def _load_raw_yaml() -> dict:
    presets_path = Path(__file__).with_name("styles.yaml")
    with presets_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_presets() -> Dict[str, StylePreset]:
    global _PRESETS_CACHE
    if _PRESETS_CACHE is not None:
        return _PRESETS_CACHE

    data = _load_raw_yaml()
    presets: Dict[str, StylePreset] = {}
    for item in data.get("presets", []):
        preset = StylePreset(
            id=item["id"],
            name=item.get("name", item["id"].title()),
            style_prefix=item.get("style_prefix", ""),
            negative_prompt=item.get("negative_prompt", ""),
        )
        presets[preset.id] = preset

    _PRESETS_CACHE = presets
    return presets


def list_presets() -> List[StylePreset]:
    return list(load_presets().values())


def get_preset(preset_id: str) -> Optional[StylePreset]:
    return load_presets().get(preset_id)


def compose_prompt(base_prompt: str, preset: Optional[StylePreset]) -> str:
    """Compose a full prompt by combining a style prefix with the user's base prompt."""

    base_prompt = base_prompt.strip()
    if not preset or not preset.style_prefix:
        return base_prompt

    # Simple heuristic: put style prefix first so the model strongly respects style.
    if base_prompt:
        return f"{preset.style_prefix}, {base_prompt}"
    return preset.style_prefix


def build_negative_prompt(
    base_negative: str,
    preset: Optional[StylePreset],
) -> str:
    """Combine global negative prompt text with preset-specific templates."""

    parts = []
    if preset and preset.negative_prompt:
        parts.append(preset.negative_prompt)
    base_negative = base_negative.strip()
    if base_negative:
        parts.append(base_negative)

    return ", ".join(parts)


__all__ = [
    "StylePreset",
    "load_presets",
    "list_presets",
    "get_preset",
    "compose_prompt",
    "build_negative_prompt",
]

