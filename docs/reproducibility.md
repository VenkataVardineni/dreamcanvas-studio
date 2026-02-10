# Reproducibility Guarantee

DreamCanvas Studio is designed so that **every generated image can be recreated** from stored metadata.

For each generation, the app stores a JSON sidecar file alongside the PNG image with at least the following fields:

- `prompt`
- `negative_prompt`
- `seed` (and `base_seed` for batches)
- `steps`
- `guidance_scale`
- `height` / `width`
- `model_id`
- `device`
- `duration_sec`

Re-running the Stable Diffusion pipeline with the same values for these fields will, as closely as possible, reproduce the original output. The Streamlit UI provides a **Reproduce** button in the gallery detail view that automates this process.

