from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image


OUTPUT_ROOT = Path("outputs")


@dataclass
class GenerationRecord:
    id: str
    created_at: str
    image_path: str
    metadata_path: str
    prompt: str
    negative_prompt: str
    preset_id: Optional[str]
    seed: int
    steps: int
    guidance_scale: float
    model_id: str
    device: str
    duration_sec: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _today_dir() -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = OUTPUT_ROOT / today
    _ensure_dir(out_dir)
    return out_dir


def save_generation(
    image: Image.Image,
    metadata: Dict[str, Any],
    preset_id: Optional[str] = None,
) -> GenerationRecord:
    """Persist a single generation as PNG + JSON sidecar.

    Layout: outputs/<YYYY-MM-DD>/<id>.png and <id>.json
    """

    out_dir = _today_dir()

    created_at = datetime.now().isoformat(timespec="seconds")
    # Use timestamp-based ID for simplicity and human readability.
    base_id = datetime.now().strftime("%H%M%S%f")
    file_stem = f"{base_id}"

    img_path = out_dir / f"{file_stem}.png"
    json_path = out_dir / f"{file_stem}.json"

    # Save image
    image.save(img_path, format="PNG")

    # Merge metadata with storage fields
    stored_metadata = {
        **metadata,
        "id": file_stem,
        "created_at": created_at,
        "image_path": str(img_path),
        "preset_id": preset_id,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stored_metadata, f, indent=2)

    record = GenerationRecord(
        id=file_stem,
        created_at=created_at,
        image_path=str(img_path),
        metadata_path=str(json_path),
        prompt=stored_metadata.get("prompt", ""),
        negative_prompt=stored_metadata.get("negative_prompt", ""),
        preset_id=preset_id,
        seed=int(stored_metadata.get("seed", 0)),
        steps=int(stored_metadata.get("steps", 0)),
        guidance_scale=float(stored_metadata.get("guidance_scale", 0.0)),
        model_id=stored_metadata.get("model_id", ""),
        device=stored_metadata.get("device", ""),
        duration_sec=float(stored_metadata.get("duration_sec", 0.0)),
    )

    return record


def _iter_metadata_files() -> Iterable[Path]:
    if not OUTPUT_ROOT.exists():
        return []
    for day_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not day_dir.is_dir():
            continue
        for json_path in sorted(day_dir.glob("*.json")):
            yield json_path


def load_record(json_path: Path) -> GenerationRecord:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = Path(data["image_path"])
    return GenerationRecord(
        id=str(data["id"]),
        created_at=str(data.get("created_at", "")),
        image_path=str(image_path),
        metadata_path=str(json_path),
        prompt=data.get("prompt", ""),
        negative_prompt=data.get("negative_prompt", ""),
        preset_id=data.get("preset_id"),
        seed=int(data.get("seed", 0)),
        steps=int(data.get("steps", 0)),
        guidance_scale=float(data.get("guidance_scale", 0.0)),
        model_id=data.get("model_id", ""),
        device=data.get("device", ""),
        duration_sec=float(data.get("duration_sec", 0.0)),
    )


def list_generations(
    *,
    preset_id: Optional[str] = None,
    date: Optional[str] = None,
    keyword: Optional[str] = None,
) -> List[GenerationRecord]:
    """List generations with optional filters by preset, date, and prompt keyword."""

    records: List[GenerationRecord] = []

    for json_path in _iter_metadata_files():
        if date and json_path.parent.name != date:
            continue

        rec = load_record(json_path)

        if preset_id and rec.preset_id != preset_id:
            continue
        if keyword and keyword.lower() not in rec.prompt.lower():
            continue

        records.append(rec)

    # Most recent first
    records.sort(key=lambda r: (r.created_at, r.id), reverse=True)
    return records


def load_image(record: GenerationRecord) -> Image.Image:
    return Image.open(record.image_path)


def record_to_dict(record: GenerationRecord) -> Dict[str, Any]:
    return asdict(record)


__all__ = [
    "GenerationRecord",
    "save_generation",
    "list_generations",
    "load_image",
    "record_to_dict",
]

