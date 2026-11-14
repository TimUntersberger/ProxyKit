"""Make a strict 2x FLUX composite and a same-scale Vivi comparison."""

from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
parser = ArgumentParser()
parser.add_argument("--variant", choices=("single", "maskref"), default="maskref")
variant = parser.parse_args().variant
base = np.asarray(Image.open(HERE / "batch_OP13-012_SP_sharp_palette_workflow.png").convert("RGB"))
candidate = np.asarray(Image.open(HERE / f"vivi_flux2_klein_{variant}_2x.png").convert("RGB"))
glyph = np.asarray(Image.open(
    HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png"
).convert("L")) > 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
ring = (cv2.dilate(glyph.astype(np.uint8), kernel) > 0) & ~(
    cv2.erode(glyph.astype(np.uint8), kernel) > 0
)
selection = cv2.resize(ring.astype(np.uint8), (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
weight = np.clip(distance / 8, 0, 1) * selection
result = np.uint8(np.clip(
    base.astype(np.float32) * (1 - weight[..., None])
    + candidate.astype(np.float32) * weight[..., None], 0, 255,
))
result[~selection] = base[~selection]
output = HERE / f"batch_OP13-012_SP_flux2_klein_{variant}_full_2x.png"
Image.fromarray(result).save(output)
print(output.name, "changed", int(np.count_nonzero(np.any(result != base, axis=2))),
      "outside_mask", int(np.count_nonzero(np.any(result != base, axis=2) & ~selection)))

paths = [
    HERE / "batch_OP13-012_SP_sharp_palette_workflow.png",
    HERE / "batch_OP13-012_SP_outline10_full_2x.png",
    HERE / "batch_OP13-012_SP_cloud_same_mask_1.0_full_2x.png",
    output,
]
labels = ["Source repair", "SD 1.5", "Cloud edit", f"FLUX.2 Klein {variant}"]
box = (0, 650, 1280, 1090)
panel_width = (box[2] - box[0]) // 2
panel_height = (box[3] - box[1]) // 2
sheet = Image.new("RGB", (4 * panel_width, panel_height + 42), "#f3f3f3")
draw = ImageDraw.Draw(sheet)
for index, (path, label) in enumerate(zip(paths, labels)):
    crop = Image.open(path).convert("RGB").crop(box).resize(
        (panel_width, panel_height), Image.Resampling.LANCZOS)
    sheet.paste(crop, (index * panel_width, 42))
    draw.text((index * panel_width + 12, 12), label, fill="#111111")
comparison = HERE / f"vivi_flux2_klein_{variant}_scroll_comparison.png"
sheet.save(comparison)
print(comparison.name)
