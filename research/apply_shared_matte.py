"""Estimate a common white overlay from the watermarked reference set."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

HERE = Path(__file__).parent
SOURCE = Path(r"C:\Users\timun\Downloads\OP18-085.webp")
source = Image.open(SOURCE).convert("RGB")
rgb = np.asarray(source, dtype=np.float32)
q10 = np.asarray(Image.open(HERE / "shared_percentile_25.png").convert("RGB"), dtype=np.float32).mean(axis=2)
font = Image.open(HERE / "approx_mask.png").convert("L").point(lambda x: 255 if x > 5 else 0)
coarse = np.asarray(font.filter(ImageFilter.MaxFilter(31)), dtype=np.float32) / 255.0

# Interpolate the low-percentile background from pixels outside the shared letter shape.
valid = 1.0 - coarse
blurred_values = np.asarray(Image.fromarray(np.clip(q10 * valid, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(28)), dtype=np.float32)
blurred_weights = np.asarray(Image.fromarray((valid * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(28)), dtype=np.float32) / 255.0
background = blurred_values / np.maximum(blurred_weights, .001)
alpha_raw = np.clip((q10 - background) / np.maximum(255.0 - background, 1.0), 0.0, .95) * coarse
alpha = np.where(alpha_raw > .11, alpha_raw, 0)
alpha = np.asarray(Image.fromarray(np.clip(alpha * 255, 0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(.7)), dtype=np.float32) / 255.0
alpha[:345] = 0
alpha[498:] = 0
alpha[:, :45] = 0
alpha[:, 555:] = 0
Image.fromarray(np.clip(alpha * 255, 0, 255).astype("uint8")).save(HERE / "shared_alpha_matte.png")
print("estimated alpha core quantiles", np.quantile(alpha[alpha > .2], [.1,.5,.9]).round(3).tolist())

for gain in (.75, .9, 1.0, 1.1, 1.25):
    a = np.clip(alpha * gain, 0, .9)
    restored = np.clip((rgb - a[..., None] * 255.0) / (1 - a[..., None]), 0, 255).astype("uint8")
    Image.fromarray(restored).save(HERE / f"shared_inverse_{gain:.2f}.png")

for cutoff in (.15, .25, .35):
    binary = Image.fromarray(np.where(alpha > cutoff, 255, 0).astype("uint8"))
    feathered = np.asarray(binary.filter(ImageFilter.GaussianBlur(.65)), dtype=np.float32) / 255.0
    a = feathered * .65
    restored = np.clip((rgb - a[..., None] * 255.0) / (1 - a[..., None]), 0, 255).astype("uint8")
    Image.fromarray(restored).save(HERE / f"shared_binary_{cutoff:.2f}.png")

binary = Image.fromarray(np.where(alpha > .25, 255, 0).astype("uint8"))
for dx in (-2, 0, 2):
    for dy in (0, 2, 4, 6):
        shifted = Image.new("L", binary.size, 0)
        shifted.paste(binary, (dx, dy))
        feathered = np.asarray(shifted.filter(ImageFilter.GaussianBlur(.65)), dtype=np.float32) / 255
        a = feathered * .65
        restored = np.clip((rgb - a[...,None] * 255) / (1-a[...,None]), 0, 255).astype("uint8")
        Image.fromarray(restored).save(HERE / f"shared_shift_{dx:+d}_{dy:+d}.png")
