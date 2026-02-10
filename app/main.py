from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

from src.generation.generate import ALLOWED_RESOLUTIONS, ResolutionError, generate_batch
from src.presets import (
    StylePreset,
    build_negative_prompt,
    compose_prompt,
    get_preset,
    list_presets,
)
from src.storage.store import (
    GenerationRecord,
    list_generations,
    load_image,
    record_to_dict,
    save_generation,
)


st.set_page_config(
    page_title="DreamCanvas Studio",
    layout="wide",
)


def _init_state() -> None:
    if "selected_preset_id" not in st.session_state:
        st.session_state.selected_preset_id = None
    if "gallery_filter_preset" not in st.session_state:
        st.session_state.gallery_filter_preset = None
    if "gallery_filter_keyword" not in st.session_state:
        st.session_state.gallery_filter_keyword = ""
    if "selected_record_id" not in st.session_state:
        st.session_state.selected_record_id = None
    if "compare_selection" not in st.session_state:
        st.session_state.compare_selection = []


def sidebar_controls() -> Dict[str, Any]:
    presets = list_presets()
    preset_options = ["None"] + [p.name for p in presets]
    preset_map = {p.name: p.id for p in presets}

    st.sidebar.markdown("## Model & Settings")

    model_id = st.sidebar.text_input(
        "Model ID",
        value="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model id to load via diffusers.",
    )

    steps = st.sidebar.slider("Steps", min_value=10, max_value=60, value=30, step=5)
    guidance = st.sidebar.slider("Guidance scale", min_value=1.0, max_value=15.0, value=7.5, step=0.5)
    batch_size = st.sidebar.slider("Batch size", min_value=1, max_value=4, value=1, step=1)

    res_label = st.sidebar.selectbox(
        "Resolution",
        [f"{h} x {w}" for (h, w) in ALLOWED_RESOLUTIONS],
        index=0,
    )
    h_str, w_str = res_label.split(" x ")
    height, width = int(h_str), int(w_str)

    seed = st.sidebar.number_input("Seed (base)", min_value=0, value=42, step=1)

    st.sidebar.markdown("## Style preset")
    preset_choice = st.sidebar.selectbox("Preset", options=preset_options, index=0)

    selected_preset_id: Optional[str]
    if preset_choice == "None":
        selected_preset_id = None
    else:
        selected_preset_id = preset_map.get(preset_choice)

    st.session_state.selected_preset_id = selected_preset_id

    return {
        "model_id": model_id,
        "steps": steps,
        "guidance": guidance,
        "batch_size": batch_size,
        "height": height,
        "width": width,
        "seed": seed,
        "selected_preset_id": selected_preset_id,
    }


def prompt_section(settings: Dict[str, Any]) -> Dict[str, Any]:
    st.markdown("## Prompt")

    base_prompt = st.text_area("Prompt", value="an elegant product photo of a wireless headphone on a marble table", height=100)
    base_negative = st.text_area(
        "Negative prompt",
        value="text, watermark, logo, low resolution, blurry",
        height=60,
    )

    preset: Optional[StylePreset] = None
    if settings["selected_preset_id"]:
        preset = get_preset(settings["selected_preset_id"])

    composed_prompt = compose_prompt(base_prompt, preset)
    composed_negative = build_negative_prompt(base_negative, preset)

    st.markdown("### Effective prompt")
    st.code(composed_prompt or "(empty)")

    if composed_negative:
        st.markdown("### Effective negative prompt")
        st.code(composed_negative)

    col_gen, col_info = st.columns([1, 2])
    with col_gen:
        generate_clicked = st.button("Generate", type="primary")

    info = {
        "base_prompt": base_prompt,
        "base_negative": base_negative,
        "composed_prompt": composed_prompt,
        "composed_negative": composed_negative,
        "generate_clicked": generate_clicked,
        "preset": preset,
    }

    return info


def _do_generate(
    settings: Dict[str, Any],
    prompt_info: Dict[str, Any],
) -> List[GenerationRecord]:
    records: List[GenerationRecord] = []

    try:
        with st.spinner("Generating images..."):
            images, metas = generate_batch(
                prompt_info["composed_prompt"],
                negative_prompt=prompt_info["composed_negative"],
                base_seed=settings["seed"],
                num_images=settings["batch_size"],
                num_inference_steps=settings["steps"],
                guidance_scale=settings["guidance"],
                height=settings["height"],
                width=settings["width"],
                model_id=settings["model_id"],
            )

        for img, meta in zip(images, metas):
            rec = save_generation(
                img,
                metadata=meta,
                preset_id=settings["selected_preset_id"],
            )
            records.append(rec)

        st.success(f"Generated {len(records)} image(s).")
    except ResolutionError as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.error(f"Generation failed: {e}")

    return records


def _gallery_filters() -> Dict[str, Any]:
    st.markdown("## Gallery filters")

    presets = list_presets()
    preset_labels = ["All"] + [p.name for p in presets]
    preset_map = {p.name: p.id for p in presets}

    selected_label = st.selectbox("Preset", options=preset_labels, index=0)
    keyword = st.text_input("Prompt keyword", value=st.session_state.gallery_filter_keyword)

    if selected_label == "All":
        preset_id = None
    else:
        preset_id = preset_map.get(selected_label)

    st.session_state.gallery_filter_preset = preset_id
    st.session_state.gallery_filter_keyword = keyword

    return {"preset_id": preset_id, "keyword": keyword}


def gallery_section() -> None:
    st.markdown("## Gallery")

    filters = _gallery_filters()
    records = list_generations(
        preset_id=filters["preset_id"],
        keyword=filters["keyword"],
    )

    if not records:
        st.info("No generations yet. Generate something to see it here.")
        return

    cols = st.columns(4)
    for idx, rec in enumerate(records):
        col = cols[idx % len(cols)]
        with col:
            try:
                img = load_image(rec)
                st.image(img, use_column_width=True)
            except Exception:  # noqa: BLE001
                st.write("(image missing)")

            st.caption(
                f"{rec.prompt[:40]}...\n"
                f"Preset: {rec.preset_id or 'None'} | Seed: {rec.seed} | {rec.duration_sec:.1f}s"
            )

            detail_col, select_col = st.columns([1, 1])
            with detail_col:
                if st.button("Details", key=f"detail_{rec.id}"):
                    st.session_state.selected_record_id = rec.id
            with select_col:
                checked = rec.id in st.session_state.compare_selection
                new_checked = st.checkbox("Compare", key=f"cmp_{rec.id}", value=checked)
                if new_checked and rec.id not in st.session_state.compare_selection:
                    st.session_state.compare_selection.append(rec.id)
                elif not new_checked and rec.id in st.session_state.compare_selection:
                    st.session_state.compare_selection.remove(rec.id)

    detail_view(records)
    compare_view(records)


def _find_record(records: List[GenerationRecord], rec_id: str) -> Optional[GenerationRecord]:
    for r in records:
        if r.id == rec_id:
            return r
    return None


def detail_view(records: List[GenerationRecord]) -> None:
    rec_id = st.session_state.selected_record_id
    if not rec_id:
        return

    rec = _find_record(records, rec_id)
    if not rec:
        return

    st.markdown("---")
    st.markdown(f"## Detail: {rec.id}")

    cols = st.columns([2, 1])
    with cols[0]:
        try:
            img = load_image(rec)
            st.image(img, use_column_width=True)
        except Exception:  # noqa: BLE001
            st.write("(image missing)")

    with cols[1]:
        st.markdown("### Parameters")
        st.json(record_to_dict(rec))

        # Export buttons: JSON sidecar and PNG download
        if st.button("Export JSON metadata", key=f"export_json_{rec.id}"):
            with open(rec.metadata_path, "r", encoding="utf-8") as f:  # type: ignore[arg-type]
                data = f.read()
            st.download_button(
                "Download metadata JSON",
                data=data,
                file_name=f"{rec.id}.json",
                mime="application/json",
            )

        try:
            img = load_image(rec)
            img_bytes = _image_to_bytes(img)
            st.download_button(
                "Download PNG",
                data=img_bytes,
                file_name=f"{rec.id}.png",
                mime="image/png",
            )
        except Exception:  # noqa: BLE001
            pass

        if st.button("Reproduce", key=f"reproduce_{rec.id}"):
            _reproduce_from_record(rec)


def _image_to_bytes(img: Image.Image) -> bytes:
    from io import BytesIO

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _reproduce_from_record(rec: GenerationRecord) -> None:
    """Re-run generation with the exact parameters used for this record."""

    try:
        with open(rec.metadata_path, "r", encoding="utf-8") as f:  # type: ignore[arg-type]
            meta = json.load(f)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to load metadata for reproduction: {e}")
        return

    prompt = meta.get("prompt", "")
    negative_prompt = meta.get("negative_prompt") or ""
    base_seed = int(meta.get("base_seed") or meta.get("seed") or 0)
    steps = int(meta.get("steps", 30))
    guidance = float(meta.get("guidance_scale", 7.5))
    height = int(meta.get("height", 512))
    width = int(meta.get("width", 512))
    model_id = meta.get("model_id", "runwayml/stable-diffusion-v1-5")

    try:
        with st.spinner("Reproducing image..."):
            images, metas = generate_batch(
                prompt,
                negative_prompt=negative_prompt,
                base_seed=base_seed,
                num_images=1,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=height,
                width=width,
                model_id=model_id,
            )
        img = images[0]
        new_meta = metas[0]
        new_rec = save_generation(img, metadata=new_meta, preset_id=rec.preset_id)

        st.success("Reproduction complete. New record added to gallery.")
        st.session_state.selected_record_id = new_rec.id
    except Exception as e:  # noqa: BLE001
        st.error(f"Reproduction failed: {e}")


def compare_view(records: List[GenerationRecord]) -> None:
    sel_ids: List[str] = st.session_state.compare_selection
    if not sel_ids:
        return

    selected_records: List[GenerationRecord] = [r for r in records if r.id in sel_ids]
    if len(selected_records) < 2:
        return

    st.markdown("---")
    st.markdown("## Compare")

    cols = st.columns(len(selected_records))
    for col, rec in zip(cols, selected_records):
        with col:
            st.markdown(f"### {rec.id}")
            try:
                img = load_image(rec)
                st.image(img, use_column_width=True)
            except Exception:  # noqa: BLE001
                st.write("(image missing)")

            st.caption(f"Seed: {rec.seed} | Preset: {rec.preset_id or 'None'}")

            st.json(
                {
                    "prompt": rec.prompt,
                    "negative_prompt": rec.negative_prompt,
                    "steps": rec.steps,
                    "guidance_scale": rec.guidance_scale,
                    "model_id": rec.model_id,
                    "device": rec.device,
                    "duration_sec": rec.duration_sec,
                }
            )


def main() -> None:
    _init_state()

    st.title("DreamCanvas Studio")
    st.caption("Local Stable Diffusion studio for Apple Silicon (MPS)")

    settings = sidebar_controls()
    prompt_info = prompt_section(settings)

    if prompt_info["generate_clicked"]:
        new_records = _do_generate(settings, prompt_info)
        if new_records:
            # Auto-select the most recent record for detail view.
            st.session_state.selected_record_id = new_records[0].id

    gallery_section()


if __name__ == "__main__":
    main()

