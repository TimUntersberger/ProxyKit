"""Upscale and strictly composite selected FLUX interior-blend trials."""

from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
UPSCAYL = Path(r"C:\Program Files\Upscayl\resources\bin\upscayl-bin.exe")
MODELS = Path(r"C:\Program Files\Upscayl\resources\models")
names = ("ring_only", "fill35", "fill65", "fill65_color")
base = np.asarray(Image.open(HERE / "batch_OP13-012_SP_flux2_klein_single_full_2x.png").convert("RGB"))
glyph = np.asarray(Image.open(
    HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png"
).convert("L")) > 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
inner = cv2.erode(glyph.astype(np.uint8), kernel) > 0
selection = cv2.resize(inner.astype(np.uint8), (base.shape[1], base.shape[0]),
                       interpolation=cv2.INTER_NEAREST) > 0
distance = cv2.distanceTransform(selection.astype(np.uint8), cv2.DIST_L2, 3)
weight = np.clip(distance / 6, 0, 1) * selection

outputs = []
for name in names:
    if name == "ring_only":
        outputs.append(HERE / "batch_OP13-012_SP_flux2_klein_single_full_2x.png")
        continue
    small = HERE / f"vivi_flux2_{name}_1x.png"
    upscaled = HERE / f"vivi_flux2_{name}_2x.png"
    if not upscaled.exists():
        subprocess.run([
            str(UPSCAYL), "-i", str(small), "-o", str(upscaled),
            "-m", str(MODELS), "-n", "upscayl-standard-4x", "-s", "2", "-f", "png",
        ], check=True, stdout=subprocess.DEVNULL)
    candidate = np.asarray(Image.open(upscaled).convert("RGB"))
    result = np.uint8(np.clip(
        base.astype(np.float32) * (1 - weight[..., None])
        + candidate.astype(np.float32) * weight[..., None], 0, 255,
    ))
    result[~selection] = base[~selection]
    path = HERE / f"batch_OP13-012_SP_flux2_{name}_full_2x.png"
    Image.fromarray(result).save(path)
    outputs.append(path)
    print(path.name,
          "changed", int(np.count_nonzero(np.any(result != base, axis=2))),
          "outside", int(np.count_nonzero(np.any(result != base, axis=2) & ~selection)))

labels = ("Outline only", "Interior 35%", "Interior 65%", "Interior 65%, color matched")
sheet = Image.new("RGB", (2560, 920), "white")
draw = ImageDraw.Draw(sheet)
for row in range(2):
    box = (0 if row == 0 else 640, 700, 640 if row == 0 else 1280, 1110)
    for index, (path, label) in enumerate(zip(outputs, labels)):
        picture = Image.open(path).convert("RGB").crop(box)
        sheet.paste(picture, (index * 640, row * 460 + 40))
        draw.text((index * 640 + 10, row * 460 + 10), label, fill="black")
comparison = HERE / "vivi_flux2_interior_2x_comparison.png"
sheet.save(comparison)
print(comparison.name)
