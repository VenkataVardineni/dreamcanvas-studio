# DreamCanvas Studio

Local Stable Diffusion studio for Apple Silicon (MPS).

## Overview

DreamCanvas Studio is a local, Mac-friendly AI image generation and prompt engineering studio. It runs Stable Diffusion on Apple Silicon (M1/M2/M3) using PyTorch with the MPS backend and provides a polished UI for:

- Prompt presets and style profiles
- Negative prompts and batch variations
- Seed control and reproducible generations
- A personal gallery with full metadata for each image
- Side-by-side comparison and reproduction of any result

## Key Features

- **Apple Silicon optimized**: Uses PyTorch MPS device for fast local inference.
- **Studio-grade controls**: Steps, guidance scale, seed, batch size, resolution guardrails.
- **Style presets**: One-click cinematic, cyberpunk, watercolor, anime, product photo, and more.
- **Prompt tools**: Style-aware prompt composer and negative prompt templates.
- **Metadata-first gallery**: Every generation is stored with prompt, seed, steps, guidance, model, device, and duration.
- **Reproducibility guarantee**: Any image in the gallery can be regenerated from stored metadata.

## Quick Start (Local)

### 1. Pre-requisites

- macOS with Apple Silicon (M1/M2/M3 recommended)
- Python 3.10+ (recommended via `pyenv` or `conda`)
- `git` and `virtualenv` (optional but recommended)

### 2. Clone and set up environment

```bash
cd "~/Canvas Studio"
# If you have not cloned from GitHub yet, initialize here or clone your GitHub repo
cd dreamcanvas-studio

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Studio

```bash
source .venv/bin/activate
streamlit run app/main.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

## Performance on Apple Silicon (MPS)

DreamCanvas Studio uses the PyTorch MPS backend to run Stable Diffusion on Apple Silicon GPUs:

- **Device selection**: The app automatically prefers `mps` when available, falling back to CPU otherwise.
- **Warmup pass**: On first run, a single-step warmup inference is executed to stabilize performance and match outputs.
- **Precision**: The pipeline is configured to avoid problematic float64 usage on MPS and use supported dtypes.

## Project Structure

- `app/` – Streamlit UI application (studio layout, gallery, compare mode).
- `src/generation/` – Stable Diffusion pipeline wrapper and generation utilities.
- `src/presets/` – Style presets, prompt composer, and negative prompt templates.
- `src/storage/` – Storage layer for images and metadata (PNG + JSON sidecars).
- `src/metrics/` – Hooks for basic performance and usage metrics.
- `models/` – Local model cache directory (ignored by git).
- `assets/` – Icons, sample prompts, static assets.
- `docs/` – Documentation, demo script, screenshots.

## Setup Instructions (Summary)

1. **Create and activate a virtual environment.**
2. **Install dependencies** with `pip install -r requirements.txt`.
3. **Ensure PyTorch MPS** is installed and working (PyTorch >= 2 with `mps` support).
4. **Run the Streamlit app** with `streamlit run app/main.py`.

For more details see `setup.md` (to be added) and `docs/` for demo and screenshots.

