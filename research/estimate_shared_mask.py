"""Explore watermark matte estimation from multiple similarly watermarked cards."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
SAMPLES = HERE / "samples" / "refs"
ids = {"003", "010", "047", "063", "066", "085", "086", "088", "089", "091", "092", "101", "106", "112"}
files = [path for path in sorted(SAMPLES.glob("*.jpg")) if path.stem[-3:] in ids]
stack = np.stack([np.asarray(Image.open(path).convert("RGB").resize((600, 838), Image.Resampling.LANCZOS), dtype=np.uint8) for path in files])
print("images", len(files))

for q in (0, 10, 25, 50):
    percentile = np.percentile(stack, q, axis=0).astype("uint8")
    Image.fromarray(percentile).save(HERE / f"shared_percentile_{q:02d}.png")

# A low percentile across many different backgrounds should expose common bright lettering.
low = np.percentile(stack, 10, axis=0).mean(axis=2).astype("uint8")
background = np.asarray(Image.fromarray(low).filter(ImageFilter.GaussianBlur(12)), dtype=np.float32)
highpass = np.clip((low.astype("float32") - background) * 3.0 + 128, 0, 255).astype("uint8")
Image.fromarray(highpass).save(HERE / "shared_highpass.png")
