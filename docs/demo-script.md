# DreamCanvas Studio – 60-second Demo Script

1. **Open the Studio**
   - Run `streamlit run app/main.py` and open the app in your browser.
   - Mention: "This is DreamCanvas Studio – a local Stable Diffusion studio optimized for Apple Silicon (MPS)."

2. **Enter a prompt and choose a style**
   - In the main prompt area, type: `a cinematic portrait of a person standing in the rain, city lights in the background`.
   - On the left sidebar, pick the **Cinematic** style preset.
   - Briefly show how the effective prompt and negative prompt update live.

3. **Adjust generation settings**
   - Highlight the controls: steps, guidance scale, batch size, seed, and resolution.
   - Set batch size to 2 and keep the default seed.

4. **Generate images**
   - Click **Generate**.
   - Point out the spinner and mention that a warmup pass has already been done for smoother performance.
   - When the images appear, note the displayed seed, preset, and generation time.

5. **Use the gallery and detail view**
   - Scroll to the gallery section.
   - Click **Details** on one of the generated images.
   - Show the full-size image and the exact parameters / metadata JSON.

6. **Reproduce an image**
   - In the detail panel, click **Reproduce**.
   - Explain that this re-runs Stable Diffusion with the same seed, prompt, and settings to produce a reproducible result.
   - Point out that the new image appears in the gallery with its own record.

7. **Compare multiple results**
   - In the gallery, select 2–4 images using the **Compare** checkboxes.
   - Scroll to the **Compare** section to show them side-by-side.
   - Call out parameter differences (e.g., seeds, presets) visible below each image.

8. **Export metadata and image**
   - In the detail view of one image, show the **Download PNG** and **Download metadata JSON** actions.
   - Explain: "Every image can be regenerated from its stored JSON metadata – this is our reproducibility guarantee."

