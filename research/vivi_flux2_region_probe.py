"""Compare precise versus expanded repaint regions using one FLUX generation."""

from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
UPSCAYL = Path(r"C:\Program Files\Upscayl\resources\bin\upscayl-bin.exe")
MODELS = Path(r"C:\Program Files\Upscayl\resources\models")
BOX = (0, 190, 640, 702)
base_1x = np.asarray(Image.open(
    HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_palette_1x.png"
).convert("RGB"))
base_2x = np.asarray(Image.open(
    HERE / "batch_OP13-012_SP_sharp_palette_workflow.png"
).convert("RGB"))
model = np.asarray(Image.open(HERE / "vivi_flux2_klein_single_raw.png").convert("RGB"))
glyph = np.asarray(Image.open(
    HERE / "runs/batch_OP13-012_SP_eb9b91ee/repaired_1x_mask.png"
).convert("L")) > 0
outer = cv2.dilate(glyph.astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
ys, xs = np.where(glyph)
band = np.zeros_like(outer)
pad = 16
band[max(0, ys.min() - pad):min(glyph.shape[0], ys.max() + pad + 1),
     max(0, xs.min() - pad):min(glyph.shape[1], xs.max() + pad + 1)] = 1
masks = {
    "outer100": outer,
    "close12": cv2.morphologyEx(
        outer, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))),
    "close20": cv2.morphologyEx(
        outer, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))),
    "band100": band,
}

outputs = []
for name, mask in masks.items():
    selection = mask > 0
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    weight = np.clip(dist / 12, 0, 1) * selection
    small = base_1x.copy()
    region = small[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    local_selection = selection[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    local_weight = weight[BOX[1]:BOX[3], BOX[0]:BOX[2]]
    blended = np.uint8(np.clip(
        region.astype(np.float32) * (1 - local_weight[..., None])
        + model.astype(np.float32) * local_weight[..., None], 0, 255,
    ))
    region[local_selection] = blended[local_selection]
    small_path = HERE / f"vivi_flux2_{name}_region_1x.png"
    Image.fromarray(small).save(small_path)

    upscaled = HERE / f"vivi_flux2_{name}_region_2x.png"
    if not upscaled.exists():
        subprocess.run([
            str(UPSCAYL), "-i", str(small_path), "-o", str(upscaled),
            "-m", str(MODELS), "-n", "upscayl-standard-4x", "-s", "2", "-f", "png",
        ], check=True, stdout=subprocess.DEVNULL)
    candidate = np.asarray(Image.open(upscaled).convert("RGB"))
    selection_2x = cv2.resize(mask, (base_2x.shape[1], base_2x.shape[0]),
                              interpolation=cv2.INTER_NEAREST) > 0
    dist_2x = cv2.distanceTransform(selection_2x.astype(np.uint8), cv2.DIST_L2, 3)
    weight_2x = np.clip(dist_2x / 16, 0, 1) * selection_2x
    full = np.uint8(np.clip(
        base_2x.astype(np.float32) * (1 - weight_2x[..., None])
        + candidate.astype(np.float32) * weight_2x[..., None], 0, 255,
    ))
    full[~selection_2x] = base_2x[~selection_2x]
    path = HERE / f"batch_OP13-012_SP_flux2_{name}_region_full_2x.png"
    Image.fromarray(full).save(path)
    outputs.append(path)
    print(name, "selected_1x", int(selection.sum()),
          "changed_2x", int(np.count_nonzero(np.any(full != base_2x, axis=2))),
          "outside", int(np.count_nonzero(np.any(full != base_2x, axis=2) & ~selection_2x)))

sheet = Image.new("RGB", (2560, 920), "white")
draw = ImageDraw.Draw(sheet)
for row in range(2):
    box = (0 if row == 0 else 640, 700, 640 if row == 0 else 1280, 1110)
    for index, (name, path) in enumerate(zip(masks, outputs)):
        sheet.paste(Image.open(path).convert("RGB").crop(box),
                    (index * 640, row * 460 + 40))
        draw.text((index * 640 + 10, row * 460 + 10), name, fill="black")
comparison = HERE / "vivi_flux2_region_2x_comparison.png"
sheet.save(comparison)
print(comparison.name)

summary_paths = (
    HERE / "batch_OP13-012_SP_flux2_klein_single_full_2x.png",
    outputs[0],
    HERE / "batch_OP13-012_SP_flux2_band100_full_2x.png",
    HERE / "batch_OP13-012_SP_flux2_band100_color_full_2x.png",
)
summary_labels = ("Outline only", "Full glyph + 10px", "Full band", "Full band, color matched")
summary = Image.new("RGB", (2560, 920), "white")
summary_draw = ImageDraw.Draw(summary)
for row in range(2):
    box = (0 if row == 0 else 640, 700, 640 if row == 0 else 1280, 1110)
    for index, (path, label) in enumerate(zip(summary_paths, summary_labels)):
        summary.paste(Image.open(path).convert("RGB").crop(box),
                      (index * 640, row * 460 + 40))
        summary_draw.text((index * 640 + 10, row * 460 + 10), label, fill="black")
summary_path = HERE / "vivi_flux2_repaint_strategy_comparison.png"
summary.save(summary_path)
print(summary_path.name)
