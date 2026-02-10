# DreamCanvas Studio Setup

This guide walks you through setting up DreamCanvas Studio on macOS with Apple Silicon (M1/M2/M3).

## 1. Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10 or newer
- `git`
- (Recommended) `pyenv` or `conda` for Python environment management
- (Optional) Docker, if you want to run the app in a container (CPU-only)

## 2. Clone the repository

```bash
cd "~/Canvas Studio"
# If not already cloned:
# git clone https://github.com/<your-user>/dreamcanvas-studio.git
cd dreamcanvas-studio
```

## 3. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure you install a PyTorch build with MPS support (PyTorch 2.x or newer). On Apple Silicon, follow the official PyTorch install commands for macOS.

## 5. Run the Studio

```bash
source .venv/bin/activate
streamlit run app/main.py
```

Streamlit will print a URL (usually `http://localhost:8501`). Open it in your browser.

## 6. Docker (optional)

You can also run DreamCanvas Studio in Docker for reproducible packaging. Note that MPS acceleration is **not** available inside the Linux container; it will run on CPU.

Build the image:

```bash
docker build -t dreamcanvas-studio .
```

Run the container:

```bash
docker run --rm -p 8501:8501 dreamcanvas-studio
```

Then open `http://localhost:8501` in your browser.

