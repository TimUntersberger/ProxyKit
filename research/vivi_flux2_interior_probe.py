"""Test general FLUX color blending inside watermark glyphs, without model reruns."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
BOX = (0, 190, 640, 702)
base_full = np.asarray(Image.open(
    HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_palette_1x.png"
).convert("RGB"))
base = base_full[BOX[1]:BOX[3], BOX[0]:BOX[2]].astype(np.float32)
raw = np.asarray(Image.open(HERE / "vivi_flux2_klein_single_raw.png").convert("RGB"), np.float32)
glyph = (np.asarray(Image.open(
    HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png"
).convert("L")) > 0)[BOX[1]:BOX[3], BOX[0]:BOX[2]]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
outer = cv2.dilate(glyph.astype(np.uint8), kernel) > 0
inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
ring = outer & ~inner
selection = outer

# Estimate the model's local color offset in clean, low-gradient image regions.
gray_base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_RGB2GRAY)
gray_raw = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_RGB2GRAY)
grad_base = cv2.magnitude(cv2.Sobel(gray_base, cv2.CV_32F, 1, 0),
                          cv2.Sobel(gray_base, cv2.CV_32F, 0, 1))
grad_raw = cv2.magnitude(cv2.Sobel(gray_raw, cv2.CV_32F, 1, 0),
                         cv2.Sobel(gray_raw, cv2.CV_32F, 0, 1))
valid = (~outer) & (grad_base < 25) & (grad_raw < 25)
valid &= np.max(np.abs(base - raw), axis=2) < 55
valid_float = valid.astype(np.float32)
denom = cv2.GaussianBlur(valid_float, (0, 0), 22) + 1e-3
offset = np.stack([
    cv2.GaussianBlur((base[..., channel] - raw[..., channel]) * valid_float,
                     (0, 0), 22) / denom
    for channel in range(3)
], axis=2)
offset = np.clip(offset, -18, 18)
harmonized = np.clip(raw + offset, 0, 255)

ring_dist = cv2.distanceTransform(ring.astype(np.uint8), cv2.DIST_L2, 3)
ring_weight = np.clip(ring_dist / 5, 0, 1) * ring
inner_dist = cv2.distanceTransform((~ring).astype(np.uint8), cv2.DIST_L2, 3)
base_low = cv2.GaussianBlur(base, (0, 0), 3)
model_low = cv2.GaussianBlur(harmonized, (0, 0), 3)
detail = np.clip(model_low + 0.8 * (base - base_low), 0, 255)

cases = {
    "ring_only": (raw, 0.0),
    "fill35": (raw, 0.35),
    "fill65": (raw, 0.65),
    "fill65_color": (harmonized, 0.65),
    "fill65_detail": (detail, 0.65),
}

for name, (model, fill_alpha) in cases.items():
    weight = ring_weight.copy()
    if fill_alpha:
        transition = fill_alpha + (1 - fill_alpha) * np.exp(-inner_dist / 4)
        weight[inner] = transition[inner]
    result_crop = np.uint8(np.clip(base * (1 - weight[..., None])
                                   + model * weight[..., None], 0, 255))
    result = base_full.copy()
    result_region = result[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    result_region[selection] = result_crop[selection]
    path = HERE / f"vivi_flux2_{name}_1x.png"
    Image.fromarray(result).save(path)
    print(path.name, "selection", int(selection.sum()),
          "changed", int(np.count_nonzero(np.any(result_region != base, axis=2) & selection)),
          "outside", int(np.count_nonzero(np.any(result_region != base, axis=2) & ~selection)))

names = ["ring_only", "fill35", "fill65", "fill65_color", "fill65_detail"]
labels = ["Outline only", "Fill 35%", "Fill 65%", "Fill 65% + color", "Fill 65% + detail"]
crop_box = (0, 330, 640, 580)
sheet = Image.new("RGB", (len(names) * 640, 282), "white")
draw = ImageDraw.Draw(sheet)
for index, (name, label) in enumerate(zip(names, labels)):
    picture = Image.open(HERE / f"vivi_flux2_{name}_1x.png").crop(crop_box)
    sheet.paste(picture, (index * 640, 32))
    draw.text((index * 640 + 10, 10), label, fill="black")
sheet.save(HERE / "vivi_flux2_interior_comparison.png")
