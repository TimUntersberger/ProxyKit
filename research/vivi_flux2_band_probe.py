"""Test a watermark-bounding-band FLUX composite with color harmonization."""

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
model = np.asarray(Image.open(HERE / "vivi_flux2_klein_single_raw.png").convert("RGB"), np.float32)
glyph = (np.asarray(Image.open(
    HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png"
).convert("L")) > 0)[BOX[1]:BOX[3], BOX[0]:BOX[2]]

ys, xs = np.where(glyph)
pad = 16
x0, x1 = max(0, xs.min() - pad), min(glyph.shape[1], xs.max() + pad + 1)
y0, y1 = max(0, ys.min() - pad), min(glyph.shape[0], ys.max() + pad + 1)
band = np.zeros(glyph.shape, np.uint8)
band[y0:y1, x0:x1] = 1
distance = cv2.distanceTransform(band, cv2.DIST_L2, 3)
feather = np.clip(distance / 16, 0, 1)

# The unedited surroundings provide a local color reference for the model crop.
gray_base = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_RGB2GRAY)
gray_model = cv2.cvtColor(model.astype(np.uint8), cv2.COLOR_RGB2GRAY)
grad_base = cv2.magnitude(cv2.Sobel(gray_base, cv2.CV_32F, 1, 0),
                          cv2.Sobel(gray_base, cv2.CV_32F, 0, 1))
grad_model = cv2.magnitude(cv2.Sobel(gray_model, cv2.CV_32F, 1, 0),
                           cv2.Sobel(gray_model, cv2.CV_32F, 0, 1))
valid = (band == 0) & (grad_base < 25) & (grad_model < 25)
valid &= np.max(np.abs(base - model), axis=2) < 55
valid_float = valid.astype(np.float32)
denom = cv2.GaussianBlur(valid_float, (0, 0), 30) + 1e-3
offset = np.stack([
    cv2.GaussianBlur((base[..., channel] - model[..., channel]) * valid_float,
                     (0, 0), 30) / denom
    for channel in range(3)
], axis=2)
harmonized = np.clip(model + np.clip(offset, -18, 18), 0, 255)

cases = {
    "band75": (model, 0.75),
    "band100": (model, 1.0),
    "band75_color": (harmonized, 0.75),
    "band100_color": (harmonized, 1.0),
}
for name, (candidate, strength) in cases.items():
    weight = feather * strength
    result_crop = np.uint8(np.clip(base * (1 - weight[..., None])
                                   + candidate * weight[..., None], 0, 255))
    result = base_full.copy()
    region = result[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    region[band > 0] = result_crop[band > 0]
    path = HERE / f"vivi_flux2_{name}_1x.png"
    Image.fromarray(result).save(path)
    print(path.name, "bbox", (int(x0), int(y0 + BOX[1]), int(x1), int(y1 + BOX[1])),
          "selected", int(band.sum()),
          "changed", int(np.count_nonzero(np.any(region != base, axis=2) & (band > 0))),
          "outside", int(np.count_nonzero(np.any(region != base, axis=2) & (band == 0))))

names = tuple(cases)
sheet = Image.new("RGB", (640 * len(names), 320), "white")
draw = ImageDraw.Draw(sheet)
for index, name in enumerate(names):
    picture = Image.open(HERE / f"vivi_flux2_{name}_1x.png").crop((0, 300, 640, 580))
    sheet.paste(picture, (index * 640, 40))
    draw.text((index * 640 + 10, 10), name, fill="black")
sheet.save(HERE / "vivi_flux2_band_1x_comparison.png")
